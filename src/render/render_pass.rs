use crate::{
    EguiRenderOutput, EguiUserTextures, RenderComputedScaleFactor,
    render::{EguiBevyPaintCallback, EguiViewTarget},
};
use bevy_ecs::{query::QueryState, world::World};
use bevy_image::BevyDefault as _;
use bevy_platform::collections::HashMap;
use bevy_render::{
    camera::ExtractedCamera,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    texture::GpuImage,
    view::{ExtractedView, ViewTarget},
};
use std::sync::Mutex;
use wgpu::TextureFormat;

/// Egui pass node.
pub struct EguiPassNode {
    egui_view_query: QueryState<(
        bevy_ecs::entity::Entity,
        &'static ExtractedView,
        &'static EguiViewTarget,
        &'static EguiRenderOutput,
        &'static RenderComputedScaleFactor,
    )>,
    egui_view_target_query: QueryState<(&'static ViewTarget, &'static ExtractedCamera)>,
    /// One renderer per texture format (HDR vs SDR).
    renderers: Mutex<HashMap<TextureFormat, egui_wgpu::Renderer>>,
    /// Maps bevy_egui user texture u64 ids to egui TextureIds registered with egui-wgpu.
    user_texture_map: Mutex<HashMap<u64, egui::TextureId>>,
}

impl EguiPassNode {
    /// Creates an Egui pass node.
    pub fn new(world: &mut World) -> Self {
        Self {
            egui_view_query: world.query_filtered(),
            egui_view_target_query: world.query(),
            renderers: Mutex::new(HashMap::default()),
            user_texture_map: Mutex::new(HashMap::default()),
        }
    }

    fn get_or_create_renderer<'a>(
        renderers: &'a mut HashMap<TextureFormat, egui_wgpu::Renderer>,
        device: &wgpu::Device,
        format: TextureFormat,
    ) -> &'a mut egui_wgpu::Renderer {
        renderers.entry(format).or_insert_with(|| {
            egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default())
        })
    }
}

impl Node for EguiPassNode {
    fn update(&mut self, world: &mut World) {
        self.egui_view_query.update_archetypes(world);
        self.egui_view_target_query.update_archetypes(world);

        // Collect paint callback data (immutable world access).
        let callback_updates: Vec<_> = self
            .egui_view_query
            .iter(world)
            .flat_map(
                |(entity, _view, egui_view_target, render_output, scale_factor)| {
                    let Ok((_, camera)) =
                        self.egui_view_target_query.get(world, egui_view_target.0)
                    else {
                        return Vec::new();
                    };

                    let texture_format = if camera.hdr {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    };

                    let Some(target_size) = camera.physical_target_size else {
                        return Vec::new();
                    };

                    render_output
                        .paint_jobs
                        .iter()
                        .filter_map(|clipped| {
                            if let egui::epaint::Primitive::Callback(cb) = &clipped.primitive
                                && let Ok(callback) =
                                    cb.callback.clone().downcast::<EguiBevyPaintCallback>()
                            {
                                let info = egui::PaintCallbackInfo {
                                    viewport: cb.rect,
                                    clip_rect: clipped.clip_rect,
                                    pixels_per_point: scale_factor.scale_factor,
                                    screen_size_px: target_size.to_array(),
                                };
                                return Some((info, callback, texture_format, entity));
                            }
                            None
                        })
                        .collect::<Vec<_>>()
                },
            )
            .collect();

        // Process paint callback updates (mutable world access).
        for (info, callback, texture_format, entity) in callback_updates {
            callback
                .cb()
                .update(info, RenderEntity::from(entity), texture_format, world);
        }
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let input_view_entity = graph.view_entity();

        let Ok((_entity, _view, view_target, render_output, scale_factor)) =
            self.egui_view_query.get_manual(world, input_view_entity)
        else {
            return Ok(());
        };

        let Ok((target, camera)) = self.egui_view_target_query.get_manual(world, view_target.0)
        else {
            return Ok(());
        };

        let Some(target_size) = camera.physical_target_size else {
            return Ok(());
        };

        if target_size.x < 1 || target_size.y < 1 {
            return Ok(());
        }

        let texture_format = if camera.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let pixels_per_point = scale_factor.scale_factor;

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [target_size.x, target_size.y],
            pixels_per_point,
        };

        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let device = render_device.wgpu_device();
        let queue: &wgpu::Queue = render_queue;

        let mut renderers = self.renderers.lock().unwrap();
        let mut user_texture_map = self.user_texture_map.lock().unwrap();

        let renderer = Self::get_or_create_renderer(&mut renderers, device, texture_format);

        // 1. Process textures_delta - update/create managed textures.
        for (id, image_delta) in &render_output.textures_delta.set {
            renderer.update_texture(device, queue, *id, image_delta);
        }

        // 2. Sync user textures from Bevy assets.
        if let Some(user_textures) = world.get_resource::<EguiUserTextures>() {
            let gpu_images = world.resource::<RenderAssets<GpuImage>>();

            // Track which user texture ids are still active.
            let mut active_ids: bevy_platform::collections::HashSet<u64> =
                bevy_platform::collections::HashSet::default();

            for (asset_id, (_handle, user_id)) in &user_textures.textures {
                active_ids.insert(*user_id);

                if let Some(gpu_image) = gpu_images.get(*asset_id) {
                    let wgpu_texture_view = &*gpu_image.texture_view;

                    if let Some(egui_tex_id) = user_texture_map.get(user_id) {
                        // Update existing registration.
                        renderer.update_egui_texture_from_wgpu_texture(
                            device,
                            wgpu_texture_view,
                            wgpu::FilterMode::Linear,
                            *egui_tex_id,
                        );
                    } else {
                        // Register new texture.
                        let egui_tex_id = renderer.register_native_texture(
                            device,
                            wgpu_texture_view,
                            wgpu::FilterMode::Linear,
                        );
                        user_texture_map.insert(*user_id, egui_tex_id);
                    }
                }
            }

            // Remove textures that are no longer in the user textures list.
            user_texture_map.retain(|user_id, egui_tex_id| {
                if !active_ids.contains(user_id) {
                    renderer.free_texture(egui_tex_id);
                    false
                } else {
                    true
                }
            });
        }

        // 3. Remap paint jobs: replace User(n) texture IDs with the egui-wgpu registered IDs.
        let mut paint_jobs = render_output.paint_jobs.clone();
        for clipped in &mut paint_jobs {
            if let egui::epaint::Primitive::Mesh(mesh) = &mut clipped.primitive
                && let egui::TextureId::User(id) = mesh.texture_id
                && let Some(mapped_id) = user_texture_map.get(&id)
            {
                mesh.texture_id = *mapped_id;
            }
        }

        // 4. Filter out Bevy paint callbacks — egui-wgpu doesn't know about them
        // and would log warnings. Pass the same filtered list to both update_buffers and render.
        let mesh_jobs: Vec<_> = paint_jobs
            .iter()
            .filter(|c| matches!(c.primitive, egui::epaint::Primitive::Mesh(_)))
            .cloned()
            .collect();

        let cmd_bufs = renderer.update_buffers(
            device,
            queue,
            render_context.command_encoder(),
            &mesh_jobs,
            &screen_descriptor,
        );

        for cmd_buf in cmd_bufs {
            render_context.add_command_buffer(cmd_buf);
        }

        // 5. Render all egui meshes in one pass.
        {
            let color_attachment = target.get_unsampled_color_attachment();
            let render_pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_pass"),
                        color_attachments: &[Some(color_attachment)],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
            let mut render_pass = render_pass.forget_lifetime();
            renderer.render(&mut render_pass, &mesh_jobs, &screen_descriptor);
        }

        // 6. Handle Bevy paint callbacks in separate passes.
        for clipped in &paint_jobs {
            if let egui::epaint::Primitive::Callback(cb) = &clipped.primitive
                && let Ok(callback) = cb.callback.clone().downcast::<EguiBevyPaintCallback>()
            {
                let info = egui::PaintCallbackInfo {
                    viewport: cb.rect,
                    clip_rect: clipped.clip_rect,
                    pixels_per_point,
                    screen_size_px: [target_size.x, target_size.y],
                };

                let viewport = info.viewport_in_pixels();
                if viewport.width_px > 0 && viewport.height_px > 0 {
                    // prepare_render phase
                    callback.cb().prepare_render(
                        info,
                        render_context,
                        RenderEntity::from(input_view_entity),
                        texture_format,
                        world,
                    );

                    // Reconstruct info for the render phase.
                    let info = egui::PaintCallbackInfo {
                        viewport: cb.rect,
                        clip_rect: clipped.clip_rect,
                        pixels_per_point,
                        screen_size_px: [target_size.x, target_size.y],
                    };

                    // render phase
                    let color_attachment = target.get_unsampled_color_attachment();
                    let render_pass = render_context.command_encoder().begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("egui_paint_callback_pass"),
                            color_attachments: &[Some(color_attachment)],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        },
                    );
                    let mut render_pass = render_pass.forget_lifetime();
                    render_pass.set_viewport(
                        viewport.left_px as f32,
                        viewport.top_px as f32,
                        viewport.width_px as f32,
                        viewport.height_px as f32,
                        0.0,
                        1.0,
                    );
                    callback.cb().render(
                        info,
                        &mut render_pass,
                        RenderEntity::from(input_view_entity),
                        texture_format,
                        world,
                    );
                }
            }
        }

        // 5. Free old textures.
        for id in &render_output.textures_delta.free {
            renderer.free_texture(id);
        }

        Ok(())
    }
}
