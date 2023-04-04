//
// Implementation for Yocto/Trace.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2022 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "yocto_trace.h"

#include <math.h>

#include <algorithm>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>

#include "yocto_color.h"
#include "yocto_geometry.h"
#include "yocto_sampling.h"
#include "yocto_scene.h"
#include "yocto_shading.h"
#include "yocto_shape.h"
#include "yocto_math.h"

#ifdef YOCTO_DENOISE
#include <OpenImageDenoise/oidn.hpp>
#endif

// -----------------------------------------------------------------------------
// PARALLEL HELPERS
// -----------------------------------------------------------------------------
namespace yocto {

// Simple parallel for used since our target platforms do not yet support
// parallel algorithms. `Func` takes the two integer indices.
template <typename T, typename Func>
inline void parallel_for(T num1, T num2, Func&& func) {
  auto              futures  = vector<std::future<void>>{};
  auto              nthreads = std::thread::hardware_concurrency();
  std::atomic<T>    next_idx(0);
  std::atomic<bool> has_error(false);
  for (auto thread_id = 0; thread_id < (int)nthreads; thread_id++) {
    futures.emplace_back(std::async(
        std::launch::async, [&func, &next_idx, &has_error, num1, num2]() {
          try {
            while (true) {
              auto j = next_idx.fetch_add(1);
              if (j >= num2) break;
              if (has_error) break;
              for (auto i = (T)0; i < num1; i++) func(i, j);
            }
          } catch (...) {
            has_error = true;
            throw;
          }
        }));
  }
  for (auto& f : futures) f.get();
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF RAY-SCENE INTERSECTION
// -----------------------------------------------------------------------------
namespace yocto {

// Build the Bvh acceleration structure.
trace_bvh make_trace_bvh(const scene_data& scene, const trace_params& params) {
  if (params.embreebvh && embree_supported()) {
    return {
        {}, make_scene_ebvh(scene, params.highqualitybvh, params.noparallel)};
  } else {
    return {
        make_scene_bvh(scene, params.highqualitybvh, params.noparallel), {}};
  }
}

// Ray-intersection shortcuts
static scene_intersection intersect_scene(const trace_bvh& bvh,
    const scene_data& scene, const ray3f& ray, bool find_any = false) {
  if (bvh.ebvh.ebvh) {
    return intersect_scene_ebvh(bvh.ebvh, scene, ray, find_any);
  } else {
    return intersect_scene_bvh(bvh.bvh, scene, ray, find_any);
  }
}
static scene_intersection intersect_instance(const trace_bvh& bvh,
    const scene_data& scene, int instance, const ray3f& ray,
    bool find_any = false) {
  if (bvh.ebvh.ebvh) {
    return intersect_instance_ebvh(bvh.ebvh, scene, instance, ray, find_any);
  } else {
    return intersect_instance_bvh(bvh.bvh, scene, instance, ray, find_any);
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR PATH TRACING
// -----------------------------------------------------------------------------
namespace yocto {

// Convenience functions
[[maybe_unused]] static vec3f eval_position(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_normal(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static vec3f eval_element_normal(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_element_normal(
      scene, scene.instances[intersection.instance], intersection.element);
}
[[maybe_unused]] static vec3f eval_shading_position(const scene_data& scene,
    const scene_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_position(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec3f eval_shading_normal(const scene_data& scene,
    const scene_intersection& intersection, const vec3f& outgoing) {
  return eval_shading_normal(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv, outgoing);
}
[[maybe_unused]] static vec2f eval_texcoord(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_texcoord(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static material_point eval_material(
    const scene_data& scene, const scene_intersection& intersection) {
  return eval_material(scene, scene.instances[intersection.instance],
      intersection.element, intersection.uv);
}
[[maybe_unused]] static bool is_volumetric(
    const scene_data& scene, const scene_intersection& intersection) {
  return is_volumetric(scene, scene.instances[intersection.instance]);
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_emission(const material_point& material, const vec3f& normal,
    const vec3f& outgoing) {
  return dot(normal, outgoing) >= 0 ? material.emission : vec3f{0, 0, 0};
}

// Evaluates/sample the BRDF scaled by the cosine of the incoming direction.
static vec3f eval_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return eval_matte(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return eval_glossy(material.color, material.ior, material.roughness, normal,
        outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return eval_reflective(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::subsurface) {
    return eval_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::gltfpbr) {
    return eval_gltfpbr(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

static vec3f eval_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return eval_reflective(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return eval_transparent(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return eval_refractive(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::volumetric) {
    return eval_passthrough(material.color, normal, outgoing, incoming);
  } else {
    return {0, 0, 0};
  }
}

// Picks a direction based on the BRDF
static vec3f sample_bsdfcos(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  if (material.roughness == 0) return {0, 0, 0};

  if (material.type == material_type::matte) {
    return sample_matte(material.color, normal, outgoing, rn);
  } else if (material.type == material_type::glossy) {
    return sample_glossy(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::reflective) {
    return sample_reflective(
        material.color, material.roughness, normal, outgoing, rn);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::subsurface) {
    return sample_refractive(material.color, material.ior, material.roughness,
        normal, outgoing, rnl, rn);
  } else if (material.type == material_type::gltfpbr) {
    return sample_gltfpbr(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, rnl, rn);
  } else {
    return {0, 0, 0};
  }
}

static vec3f sample_delta(const material_point& material, const vec3f& normal,
    const vec3f& outgoing, float rnl) {
  if (material.roughness != 0) return {0, 0, 0};

  if (material.type == material_type::reflective) {
    return sample_reflective(material.color, normal, outgoing);
  } else if (material.type == material_type::transparent) {
    return sample_transparent(
        material.color, material.ior, normal, outgoing, rnl);
  } else if (material.type == material_type::refractive) {
    return sample_refractive(
        material.color, material.ior, normal, outgoing, rnl);
  } else if (material.type == material_type::volumetric) {
    return sample_passthrough(material.color, normal, outgoing);
  } else {
    return {0, 0, 0};
  }
}

// Compute the weight for sampling the BRDF
static float sample_bsdfcos_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness == 0) return 0;

  if (material.type == material_type::matte) {
    return sample_matte_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::glossy) {
    return sample_glossy_pdf(material.color, material.ior, material.roughness,
        normal, outgoing, incoming);
  } else if (material.type == material_type::reflective) {
    return sample_reflective_pdf(
        material.color, material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::subsurface) {
    return sample_refractive_pdf(material.color, material.ior,
        material.roughness, normal, outgoing, incoming);
  } else if (material.type == material_type::gltfpbr) {
    return sample_gltfpbr_pdf(material.color, material.ior, material.roughness,
        material.metallic, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

static float sample_delta_pdf(const material_point& material,
    const vec3f& normal, const vec3f& outgoing, const vec3f& incoming) {
  if (material.roughness != 0) return 0;

  if (material.type == material_type::reflective) {
    return sample_reflective_pdf(material.color, normal, outgoing, incoming);
  } else if (material.type == material_type::transparent) {
    return sample_tranparent_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::refractive) {
    return sample_refractive_pdf(
        material.color, material.ior, normal, outgoing, incoming);
  } else if (material.type == material_type::volumetric) {
    return sample_passthrough_pdf(material.color, normal, outgoing, incoming);
  } else {
    return 0;
  }
}

/*
Compute transmittance to light. Skip through index-matching shapes.
TO Be COMPLETED
*/
vec3f next_event_estimation_final(const scene_data& scene,
                          const trace_lights& lights,
                          rng_state& rng, 
                          vec3f p,
                          int bounces,
                          vec3f dir_view, // previous vertex to current point
                          bool is_surface,
                          scene_intersection vertex,
                          vector<material_point>& volume_stack,
                          int max_null_collissions,
                          const ray3f& ray_,
                          const trace_bvh& bvh,
                          int id
                           ) {
    // Sample a point on light
    // p' (p_prime) is the point on the light source
    auto ray        = ray_;
    vec2f light_uv{rand2f(rng)};
    float light_w  = rand1f(rng);
    float shape_w  = rand1f(rng);
    //int light_id = sample_light(scene, light_w)
    auto incoming = sample_lights(scene, lights, p, rand1f(rng), rand1f(rng), rand2f(rng));
    vec3f orig_point = p;

    
    int shadow_bounces = 0;
    auto opbounce      = 0;

    vec3f transmittance_light = one3f; 

    vec3f p_trans_nee = one3f;
    vec3f p_trans_dir = one3f; // for multiple importance sampling

     // Give a second check in case are needed the previus computations
    //auto ray = ray3f{p, incoming, };
    auto next_emission     = true;
    auto next_intersection = scene_intersection{};

    float next_t = float(0);
    while (true){

      auto intersection = next_emission ? intersect_scene(bvh, scene, ray)
                                      : next_intersection;

      auto vertex = intersection;
      next_t = vertex.distance;

      if (vertex.hit){
        // Give a second check
          //next_t = distance(p, next_intersection.):
      }

      // Give a second check
      if(!volume_stack.empty()){
        auto& vsdf        = volume_stack.back();
        float u  = rand1f(rng);
        int channel = clamp((int)u*3, 0, 2);
        int iteration = 0;
        float accum_t = 0;
        auto volume = vsdf.volume;
        auto  max_density = volume.max_voxel * volume.density_mult;
        auto  majorant = mean(vsdf.scattering * max_density);  // to check and implement a get_majorant function if needed NSPI
        while (true){
          if(majorant <= 0.0f) break;

          if(iteration >= max_null_collissions) break;

          float t = -log(1.0f - rand1f(rng)) / (majorant);
          float dt = next_t - accum_t;

          // Update accumulated distance
          accum_t = std::min(accum_t + t, next_t);

          if (t < dt){
            vec3f p = ray.o + incoming * accum_t;
            auto density = eval_vpt_density(vsdf.volume, p);  // Give a second check
            auto sigma_s = density * vsdf.scattering;  // Give a second check
            auto sigma_a = density * (1 - vsdf.scattering);  // Give a second check
            auto sigma_t = sigma_s + sigma_a;      // Give a second check
            auto sigma_n = majorant * (1 - sigma_t / majorant);  // Give a second check

            auto real_prob = sigma_t / majorant;

            transmittance_light *= exp(-majorant * t) * sigma_n / majorant; 
            p_trans_nee *= exp(-majorant * t) * majorant / majorant;
            p_trans_dir *= exp(-majorant * t) * majorant * (1 - real_prob) / majorant;

            if ( max(transmittance_light) <= 0 ) {
                        break;
            }
    
          }
          else{
            transmittance_light *= exp(-majorant * dt);
            p_trans_nee *= exp(-majorant * dt);
            p_trans_dir *= exp(-majorant * dt);
            break;
          }
          iteration++;
        }
      }

      if ( !vertex.hit ) {
            // Nothing is blocking, we’re done
            break;
      }
      else{
        // Something is blocking: is it an opaque surface?
        auto outgoing = -ray.d;
        auto position = eval_shading_position(scene, intersection, outgoing);
        auto normal   = eval_shading_normal(scene, intersection, outgoing);
        auto material = eval_material(scene, intersection);

        if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
          if (opbounce++ > 128) break;
          ray = {position + ray.d * 1e-2f, ray.d};
          shadow_bounces -= 1;
          // we’re blocked
          // Give a second check
          return zero3f;
          continue;
        }
        // otherwise, it’s an index-matching surface and
        // we want to pass through -- this introduces
        // one extra connection vertex
        shadow_bounces++;

        // Give a second check if this is needed
        /*
        if ( max_depth != -1 && bounces + shadow_bounces >= scene.options.max_depth ) {
                // Reach the max no. of vertices
                return make_zero_spectrum();
            }
        */
        p = p + next_t * incoming;
      }
    }
    if ( max(transmittance_light) > 0 ){
      auto pdf_dir = zero3f;
      auto evaluated_f = zero3f;

      auto intersection = next_emission ? intersect_scene(bvh, scene, ray)
                                      : next_intersection;
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      auto Le = eval_emission(material, normal, outgoing);

      // Give a second check
      float jacobian = max(-dot(incoming, normal), float(0)) /
                                distance_squared(incoming, orig_point);

      auto light_pmf = sample_discrete_pdf(lights.lights[id].elements_cdf, id);

      vec3f pdf_nee =  light_pmf * sample_lights_pdf(scene, bvh, lights, position, incoming) * p_trans_nee;

      auto material = eval_material(scene, intersection);
      if (is_surface){
        

        float pdf_bsdf = sample_bsdfcos_pdf(material, normal, outgoing, incoming);

        if (pdf_bsdf <= 0) {
                // Numerical issue -- we generated some invalid rays.
                return zero3f;
            }
        pdf_dir = pdf_bsdf * jacobian * p_trans_dir;
      }
      else{
        vec2f phase_uv{rand2f(rng)};
        float pdf_phase = sample_phasefunction_pdf(material.scanisotropy, outgoing, incoming);
        pdf_dir = pdf_phase * jacobian * p_trans_dir;
      }
      vec3f contrib = transmittance_light * evaluated_f * Le * jacobian / mean(pdf_nee);
      vec3f w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_dir * pdf_dir);

      return contrib * w;
    }
    return zero3f;
}

// NSPI: eval heterogeneus volume - TO DO: fix
std::pair<float, vec3f> eval_unidirectional_spectral_mis_NSPI(material_point& vsdf, float max_distance, rng_state& rng, const ray3f& ray) {
    auto volume           = vsdf.volume;
    auto density_vol      = volume.density_vol;
    auto emission_vol     = volume.emission_vol;
    auto max_density      = volume.max_voxel * volume.density_mult;
    auto imax_density     = 1.0f / max_density;
    auto path_length      = 0.0f;
    auto f                = one3f;
    auto p                = one3f;
    auto cc               = clamp((int)(rand1f(rng) * 3), 0, 2);
    auto current_pos      = ray.o; 
    auto majorant_density = max_density;
    auto tr               = one3f;
    
    while (true) {
      path_length     -= majorant_density == 0.0f ? -flt_max :	log(1 - rand1f(rng)) / majorant_density;
      if (path_length >= max_distance)
	    break;
      current_pos = ray.o + path_length * ray.d;
      auto d = eval_vpt_density(vsdf.volume, current_pos);
      auto sigma_t     = vec3f{d, d, d};
      auto sigma_s     = sigma_t * vsdf.scattering;
      auto sigma_a     = sigma_t - sigma_s;
      auto sigma_n     = vec3f{max_density, max_density, max_density} - sigma_t;
      vec3f sigma[3] = { sigma_n, sigma_s, sigma_a };
 
      tr *= exp(-sigma_t * path_length); 
      // Sample event 
      auto e = sample_event(sigma_a[cc] * imax_density,  sigma_s[cc] * imax_density, sigma_n[cc] * imax_density, rand1f(rng));

      if (e == material_event::null) continue;  // TO DO: double check questo if che non mi convice

      if (e == material_event::null) f *= sigma[0];
      else if (e == material_event::scatter) f *= sigma[1];
      else if (e == material_event::absorb) f *= sigma[2];
      p  = f;
      
      // Populate vsdf with medium interaction information and return
      vsdf.event = e;
      vsdf.density = sigma_t;
      f *= tr;
      f = f / mean(p);       
      break;
    }    
    return {path_length, f };
}
  
static material_event sample_event(float pa, float ps, float pn, float rn) {
  // https://stackoverflow.com/a/26751752
  auto weights = vec3f{pa, ps, pn};
  auto numbers = vector<material_event>{material_event::absorb, material_event::scatter, material_event::null};
  float sum = 0.0f;
  for (int i = 0; i < 3; ++i) {
    sum += weights[i];
    if(rn < sum) {
      //printf("weights: pa: %f\tps: %f\tpn: %f\tsum: %i\n", pa, ps, pn, numbers[i]);
      return numbers[i];
    }
  }
  // Can reach this point only if |weights| < 1 (WRONG)
  //printf("I should not be here : %f\n", pa + ps + pn);
  return material_event::null;
}



/**
// OUR CODE FOR EVALUATE VOLUME NSPI
static material_point eval_volume(const material_point& material){
  // INSERIRE CODICE NSPI
  auto volume = material_point{};

  }

}


bool has_vpt_volume(const material_point& material) {
    return (material.density != nullptr ||
            object->emission_vol != nullptr);
}
*/

/**
// Implementation of sample functione ispire by Mitsuba3 VolumeMIS NSPI
// Sampler parameter and medium are missing, trace_params used instead
static vec3f sample_volume_nspi(const scene_data& scene, const ray3f& ray_,
const trace_params& params, rng_state& rng){

}
*/

static vec3f eval_scattering(const material_point& material,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.density == vec3f{0, 0, 0}) return {0, 0, 0};
  return material.scattering * material.density *
         eval_phasefunction(material.scanisotropy, outgoing, incoming);
}

static vec3f sample_scattering(const material_point& material,
    const vec3f& outgoing, float rnl, const vec2f& rn) {
  if (material.density == vec3f{0, 0, 0}) return {0, 0, 0};
  return sample_phasefunction(material.scanisotropy, outgoing, rn);
}

static float sample_scattering_pdf(const material_point& material,
    const vec3f& outgoing, const vec3f& incoming) {
  if (material.density == vec3f{0, 0, 0}) return 0;
  return sample_phasefunction_pdf(material.scanisotropy, outgoing, incoming);
}

// Sample camera
static ray3f sample_camera(const camera_data& camera, const vec2i& ij,
    const vec2i& image_size, const vec2f& puv, const vec2f& luv, bool tent) {
  if (!tent) {
    auto uv = vec2f{
        (ij.x + puv.x) / image_size.x, (ij.y + puv.y) / image_size.y};
    return eval_camera(camera, uv, sample_disk(luv));
  } else {
    const auto width  = 2.0f;
    const auto offset = 0.5f;
    auto       fuv =
        width *
            vec2f{
                puv.x < 0.5f ? sqrt(2 * puv.x) - 1 : 1 - sqrt(2 - 2 * puv.x),
                puv.y < 0.5f ? sqrt(2 * puv.y) - 1 : 1 - sqrt(2 - 2 * puv.y),
            } +
        offset;
    auto uv = vec2f{
        (ij.x + fuv.x) / image_size.x, (ij.y + fuv.y) / image_size.y};
    return eval_camera(camera, uv, sample_disk(luv));
  }
}

// Sample lights wrt solid angle
static vec3f sample_lights(const scene_data& scene, const trace_lights& lights,
    const vec3f& position, float rl, float rel, const vec2f& ruv) {
  auto  light_id = sample_uniform((int)lights.lights.size(), rl);
  auto& light    = lights.lights[light_id];
  if (light.instance != invalidid) {
    auto& instance  = scene.instances[light.instance];
    auto& shape     = scene.shapes[instance.shape];
    auto  element   = sample_discrete(light.elements_cdf, rel);
    auto  uv        = (!shape.triangles.empty()) ? sample_triangle(ruv) : ruv;
    auto  lposition = eval_position(scene, instance, element, uv);
    return normalize(lposition - position);
  } else if (light.environment != invalidid) {
    auto& environment = scene.environments[light.environment];
    if (environment.emission_tex != invalidid) {
      auto& emission_tex = scene.textures[environment.emission_tex];
      auto  idx          = sample_discrete(light.elements_cdf, rel);
      auto  uv = vec2f{((idx % emission_tex.width) + 0.5f) / emission_tex.width,
          ((idx / emission_tex.width) + 0.5f) / emission_tex.height};
      return transform_direction(environment.frame,
          {cos(uv.x * 2 * pif) * sin(uv.y * pif), cos(uv.y * pif),
              sin(uv.x * 2 * pif) * sin(uv.y * pif)});
    } else {
      return sample_sphere(ruv);
    }
  } else {
    return {0, 0, 0};
  }
}

// Sample lights pdf
static float sample_lights_pdf(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const vec3f& position, const vec3f& direction) {
  auto pdf = 0.0f;
  for (auto& light : lights.lights) {
    if (light.instance != invalidid) {
      auto& instance = scene.instances[light.instance];
      // check all intersection
      auto lpdf          = 0.0f;
      auto next_position = position;
      for (auto bounce = 0; bounce < 100; bounce++) {
        auto intersection = intersect_instance(
            bvh, scene, light.instance, {next_position, direction});
        if (!intersection.hit) break;
        // accumulate pdf
        auto lposition = eval_position(
            scene, instance, intersection.element, intersection.uv);
        auto lnormal = eval_element_normal(
            scene, instance, intersection.element);
        // prob triangle * area triangle = area triangle mesh
        auto area = light.elements_cdf.back();
        lpdf += distance_squared(lposition, position) /
                (abs(dot(lnormal, direction)) * area);
        // continue
        next_position = lposition + direction * 1e-3f;
      }
      pdf += lpdf;
    } else if (light.environment != invalidid) {
      auto& environment = scene.environments[light.environment];
      if (environment.emission_tex != invalidid) {
        auto& emission_tex = scene.textures[environment.emission_tex];
        auto  wl = transform_direction(inverse(environment.frame), direction);
        auto  texcoord = vec2f{atan2(wl.z, wl.x) / (2 * pif),
            acos(clamp(wl.y, -1.0f, 1.0f)) / pif};
        if (texcoord.x < 0) texcoord.x += 1;
        auto i = clamp(
            (int)(texcoord.x * emission_tex.width), 0, emission_tex.width - 1);
        auto j    = clamp((int)(texcoord.y * emission_tex.height), 0,
               emission_tex.height - 1);
        auto prob = sample_discrete_pdf(
                        light.elements_cdf, j * emission_tex.width + i) /
                    light.elements_cdf.back();
        auto angle = (2 * pif / emission_tex.width) *
                     (pif / emission_tex.height) *
                     sin(pif * (j + 0.5f) / emission_tex.height);
        pdf += prob / angle;
      } else {
        pdf += 1 / (4 * pif);
      }
    }
  }
  pdf *= sample_uniform_pdf((int)lights.lights.size());
  return pdf;
}


struct trace_result {
  vec3f radiance = {0, 0, 0};
  bool  hit      = false;
  vec3f albedo   = {0, 0, 0};
  vec3f normal   = {0, 0, 0};
};

// Volumetric path tracing function NSPI
static trace_result vol_path_tracing(const scene_data& scene, const ray3f& ray_,
    const trace_params& params, const trace_lights& lights, rng_state& rng, const trace_bvh& bvh) {
  // initialize
  auto   ray      = ray_;
  auto   radiance = vec3f{0, 0, 0};
  auto   weight   = vec3f{1, 1, 1};  // (current_path_throughput )
  int    bounces  = 0;
  double dir_pdf  = 0;
  vec3f  nee_p_cache;
  double eta_scale           = 1;
  vec3f  multi_trans_pdf     = vec3f{1, 1, 1};
  int    max_null_collisions = 1000;  // to check if needs to be a parameter
  auto   hit_albedo          = vec3f{0, 0, 0};
  auto   hit_normal    = vec3f{0, 0, 0};
  auto hit           = false;

  // Give a second check in case we have more volumes
  /*
  auto volume = scene.volumes[0];
  auto density_vol = volume.density_vol;
  auto max_density = volume.max_voxel * volume.density_mult;
  */

  auto volume_stack = vector<material_point>{};

  // flag to record if nee is issued to avoid including nee pdf contribution
  // when no nee has been issued
  bool is_nee_issued = false;

  // trace  path
  // Give a second check in case is needed a while true
  while (true) {
    bool scatter      = false;
    auto intersection = intersect_scene(bvh, scene, ray);
    auto vertex       = intersection;
    auto t_hit        = INFINITY;
    if (intersection.hit) {
      t_hit = intersection.distance;
    }
    else {
      if (bounces > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    auto transmittance = vec3f{1, 1, 1};
    auto trans_dir_pdf = vec3f{1, 1, 1};
    auto trans_nee_pdf = vec3f{1, 1, 1};

    // handle transmission if inside a volume
    auto in_volume = false;

    // If there are volumes in the stack, perform volume path tracing
    if (!volume_stack.empty()) {
      /*
      auto& vsdf        = volume_stack.back();
      auto  distance = sample_transmittance(
        vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
      weight *= eval_transmittance(vsdf.density, distance) /
                sample_transmittance_pdf(
                    vsdf.density, distance, intersection.distance);
      in_volume             = distance < intersection.distance;
      intersection.distance = distance;
      */
      auto& vsdf        = volume_stack.back();
      auto volume = vsdf.volume;
      auto  max_density = volume.max_voxel * volume.density_mult;
      auto  majorant    = mean(vsdf.scattering * max_density);  // to check and implement a get_majorant function if needed NSPI

      // to get majorant[channel] if majorant is a vec3f
      auto u       = rand1f(rng);
      int  channel = clamp(int(u * 3), 0, 2);

      float accum_t   = 0;
      int   iteration = 0;


      while (true) {
        if (majorant <= 0) break;
        if (iteration >= max_null_collisions) break;

        auto t  = -log(1 - rand1f(rng)) / majorant;
        auto dt = t_hit - accum_t;
        accum_t = min(accum_t + t, t_hit);

        if (t < dt) {
          vec3f p = ray.o + ray.d * accum_t;

          auto density = eval_vpt_density(vsdf.volume, p);  // Give a second check
          auto sigma_s = density * vsdf.scattering;  // Give a second check
          auto sigma_a = density * (1 - vsdf.scattering);  // Give a second check
          auto sigma_t = sigma_s + sigma_a;      // Give a second check
          auto sigma_n = majorant * (1 - sigma_t / majorant);  // Give a second check

          auto real_prob = sigma_t / majorant;

          if (rand1f(rng) < real_prob[channel]) {
            scatter = true;
            transmittance *= exp(-majorant * t) / majorant;  // Use m exp(-majorant * t) /
                                        // max(majorant) instead if problems
            trans_dir_pdf *= exp(-majorant * t) * majorant * real_prob / majorant;  // use max instead: exp(-majorant * t) *
                                        // majorant * real_prob / max(majorant)
            ray.o = p;
            break;

          } else {
            transmittance *= exp(-majorant * t) * sigma_n / majorant;
            trans_dir_pdf *= exp(-majorant * t) * majorant * (1 - real_prob) / majorant;
            trans_nee_pdf *= exp(-majorant * t) * majorant / majorant;
          }
        } else {
          transmittance *= exp(-majorant * dt);
          trans_dir_pdf *= exp(-majorant * dt);
          trans_nee_pdf *= exp(-majorant * dt);
          //auto position = ray.o + ray.d * intersection.distance;
          auto outgoing = -ray.d;
          auto position = eval_shading_position(scene, intersection, outgoing);
          auto normal   = eval_shading_normal(scene, intersection, outgoing);
          ray.o = position;  // Give a second check in order to give a proper
                             // position as vertex.position

          break;
        }

        iteration++;
      }

      multi_trans_pdf *= trans_dir_pdf;

    } else {
      if (!intersection.hit) {
        auto position = ray.o + ray.d * intersection.distance;
        ray.o         = position;
      } else {
        return {{0, 0, 0}, false, {0, 0, 0},
            {0, 0, 0}};  // give a second check since it should be a 0 spectrum
      }
    }

    // Give a second check to this part
    //weight *= transmittance / mean(trans_dir_pdf);

    // Give a second check to this part
    // next direction 
    auto incoming = vec3f{0, 0, 0};
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);
    if (!is_delta(material)) {
        if (rand1f(rng) < 0.5f) {
          incoming = sample_bsdfcos(
              material, normal, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        if (incoming == vec3f{0, 0, 0}) break;
        weight *=
            eval_bsdfcos(material, normal, outgoing, incoming) /
            (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
                0.5f *
                    sample_lights_pdf(scene, bvh, lights, position, incoming));
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
      }

    // Hit a light source.
    // Add light contribution.
    // Fix this part by adding the id
    // If it doesn't work try with has_lights from yocto_scene.cpp
    auto id = 0;
    if (!scatter && t_hit != INFINITY && scene.materials[id].emission != vec3f({0,0,0})){
      auto Le = eval_emission(material, normal, outgoing);

      if (bounces == 0)
      {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
        // This is the only way we can see the light source, so
        // we don’t need multiple importance sampling.
        radiance += weight * Le;

        return {radiance, hit, hit_albedo, hit_normal};
      } else {
        // Need to account for next event estimation

        
        // Add light contribution only if the surface is emissive

        // Compute the probability of sampling the current intersected light point
        // from the point the next event estimation was issued 

        auto light_pmf = sample_discrete_pdf(lights.lights[id].elements_cdf, id);

        // Compute the pdf of the nee only when at least one nee has been issued
        // TODO: use is_nee_issued condition if error
        vec3f pdf_nee =  light_pmf * sample_lights_pdf(scene, bvh, lights, position, incoming) * trans_nee_pdf;

        // Next, compute the PDF for sampling the current intersected light point
        // using the latest phase function sampling + the all trasmittance sampling
        // after the last phase function sampling.


        // The geometry term (=jacobian)
        float jacobian = max(-dot(-ray.d, normal), float(0)) /
                                distance_squared(nee_p_cache, position);

        auto pdf_phase = dir_pdf * multi_trans_pdf * jacobian;

        // Compute the multi importance sampling between
        // the next event estimation and the phase function sampling
        auto w = (pdf_phase * pdf_phase) / (pdf_phase * pdf_phase + pdf_nee * pdf_nee);

        // Add the emission weighted by the multi importance sampling
        radiance += weight * Le * w;

      }
      
    }
    // Hit a index-matchcing surface
    // Give a seocond check to this part on the in_volume==material_id=-1
    if (!scatter && t_hit != INFINITY && in_volume) {
       // If the intersected surface is a index-matching surface
      // update the current medium index and 
      // pass through without scattering
      // Give a second check to this part
      //current_medium = update_medium(vertex, ray, current_medium);
      ray.o = position;
      bounces++;
      continue;

    }
    if ( bounces >= params.bounces - 1 && params.bounces != -1) break;

    if(scatter && !in_volume){
      // prepare shading point
      auto& vsdf     = volume_stack.back();

      auto density = eval_vpt_density(vsdf.volume, ray.o);  // Give a second check
      auto sigma_s = density * vsdf.scattering;  // Give a second check

      vec3f nee = next_event_estimation_final(scene,
                                                lights, 
                                                rng, 
                                                ray.o, 
                                                bounces, 
                                                -ray.d,
                                                false,
                                                vertex,
                                                volume_stack,
                                                max_null_collisions,
                                                ray,
                                                bvh,
                                                id);
                              
      radiance += weight * sigma_s * nee;
    
      // Record the last position that can issue a next event estimation
      // NEE is 0 and invalid if it is blocked by something
      // or does not reach the surface before the bounce limit
      if ( max(nee) > 0 ) {
                  nee_p_cache = ray.o;
                  is_nee_issued = true;
      }

      vec2f phase_rnd_param_uv{rand2f(rng)};

      auto next_dir_= sample_phasefunction(material.scanisotropy, outgoing, phase_rnd_param_uv);

      // Give a second check to this part in case is needed a pointer
      // Vector3 &next_dir = *next_dir_;
      vec3f next_dir = next_dir_;

      float phase_pdf = sample_phasefunction_pdf(material.scanisotropy, outgoing, incoming);

      weight *= (eval_phasefunction(material.scanisotropy, outgoing, incoming) / phase_pdf) * sigma_s;

      dir_pdf = phase_pdf;
			nee_p_cache = ray.o;
			multi_trans_pdf = one3f;
    }
    else if (t_hit != INFINITY){
      // do NEE when we hit a surface
			// what if it is a light source? We don't care we still do NEE. The radiance from the light source will
			// dominate radiance of other bounces.
			// We don't need to check for `current_medium_id != -1` because in that case scatter is set to false.
			// do next event estimation for every scatter
      vec3f nee = next_event_estimation_final(scene,
                                                lights, 
                                                rng, 
                                                ray.o, 
                                                bounces, 
                                                -ray.d,
                                                true,
                                                vertex,
                                                volume_stack,
                                                max_null_collisions,
                                                ray,
                                                bvh,
                                                id);
      radiance += weight * nee;
      // Record the last position that can issue a next event estimation
      // NEE is 0 and invalid if it is blocked by something
      // or does not reach the surface before the bounce limit
      if ( max(nee) > 0 ) {
                  nee_p_cache = ray.o;
                  is_nee_issued = true;
      }

      auto material = eval_material(scene, vertex);
      vec3f dir_view = -ray.d;
      vec2f bsdf_rnd_param_uv{rand2f(rng)};
      float bsdf_rnd_param_w = rand1f(rng);
      auto bsdf_sample_ = sample_bsdfcos(material, normal, outgoing, bsdf_rnd_param_w, bsdf_rnd_param_uv);

      if (bsdf_sample_ == zero3f) {
          // BSDF sampling failed. Abort the loop.
          break;
      }
      const auto bsdf_sample = bsdf_sample_;

      ray.d = bsdf_sample;
      // Give a second check to this part and add it if needed
      // Update ray differentials & eta_scale
      /*
      if (material.type == material_type::reflective) {
        eval_reflective(material.color, normal, outgoing, incoming);

      }
      */
      ray = {position, incoming};

      vec3f bsdf_eval = eval_bsdfcos(material, normal, outgoing, incoming);
      float pdf_bsdf = sample_bsdfcos_pdf(material, normal, outgoing, incoming);
      
      weight *= bsdf_eval / pdf_bsdf;
    }
    // russian roulette
    if (bounces > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
    bounces++;
    // Check if there is a relationship between is_lights and has_vpt_emission
    // if (has_vpt_emission(vsdf.object))
    // if ( !scatter && intersection.hit &&
    // is_light(scene.shapes[vertex.shape_id]) ) {
  }
  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing.
static trace_result trace_path(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto volume_stack  = vector<material_point>{};
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto opbounce      = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // handle transmission if inside a volume
    auto in_volume = false;
    if (!volume_stack.empty()) {
      auto& vsdf     = volume_stack.back();
      auto  distance = sample_transmittance(
          vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
      weight *= eval_transmittance(vsdf.density, distance) /
                sample_transmittance_pdf(
                    vsdf.density, distance, intersection.distance);
      in_volume             = distance < intersection.distance;
      intersection.distance = distance;
    }

    // switch between surface and volume
    if (!in_volume) {
      // prepare shading point
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      // correct roughness
      if (params.nocaustics) {
        max_roughness      = max(material.roughness, max_roughness);
        material.roughness = max_roughness;
      }

      // handle opacity
      if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
        if (opbounce++ > 128) break;
        ray = {position + ray.d * 1e-2f, ray.d};
        bounce -= 1;
        continue;
      }

      // set hit variables
      if (bounce == 0) {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
      }

      // accumulate emission
      radiance += weight * eval_emission(material, normal, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (!is_delta(material)) {
        if (rand1f(rng) < 0.5f) {
          incoming = sample_bsdfcos(
              material, normal, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        if (incoming == vec3f{0, 0, 0}) break;
        weight *=
            eval_bsdfcos(material, normal, outgoing, incoming) /
            (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
                0.5f *
                    sample_lights_pdf(scene, bvh, lights, position, incoming));
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
      }

      // update volume stack
      if (is_volumetric(scene, intersection) &&
          dot(normal, outgoing) * dot(normal, incoming) < 0) {
        if (volume_stack.empty()) {
          auto material = eval_material(scene, intersection);
          volume_stack.push_back(material);
        } else {
          volume_stack.pop_back();
        }
      }

      // setup next iteration
      ray = {position, incoming};
    } else {
      // prepare shading point
      auto  outgoing = -ray.d;
      auto  position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();

      // accumulate emission
      // radiance += weight * eval_volemission(emission, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (rand1f(rng) < 0.5f) {
        incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_scattering(vsdf, outgoing, incoming) /
          (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

      // setup next iteration
      ray = {position, incoming};
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing.
static trace_result trace_pathdirect(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray_,
    rng_state& rng, const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto volume_stack  = vector<material_point>{};
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto next_emission = true;
  auto opbounce      = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if ((bounce > 0 || !params.envhidden) && next_emission)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // handle transmission if inside a volume
    auto in_volume = false;
    if (!volume_stack.empty()) {
      auto& vsdf     = volume_stack.back();
      auto  distance = sample_transmittance(
          vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
      weight *= eval_transmittance(vsdf.density, distance) /
                sample_transmittance_pdf(
                    vsdf.density, distance, intersection.distance);
      in_volume             = distance < intersection.distance;
      intersection.distance = distance;
    }

    // switch between surface and volume
    if (!in_volume) {
      // prepare shading point
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      // correct roughness
      if (params.nocaustics) {
        max_roughness      = max(material.roughness, max_roughness);
        material.roughness = max_roughness;
      }

      // handle opacity
      if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
        if (opbounce++ > 128) break;
        ray = {position + ray.d * 1e-2f, ray.d};
        bounce -= 1;
        continue;
      }

      // set hit variables
      if (bounce == 0) {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
      }

      // accumulate emission
      if (next_emission)
        radiance += weight * eval_emission(material, normal, outgoing);

      // direct
      if (!is_delta(material)) {
        auto incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        auto pdf = sample_lights_pdf(scene, bvh, lights, position, incoming);
        auto bsdfcos = eval_bsdfcos(material, normal, outgoing, incoming);
        if (bsdfcos != vec3f{0, 0, 0} && pdf > 0) {
          auto intersection = intersect_scene(bvh, scene, {position, incoming});
          auto emission =
              !intersection.hit
                  ? eval_environment(scene, incoming)
                  : eval_emission(eval_material(scene,
                                      scene.instances[intersection.instance],
                                      intersection.element, intersection.uv),
                        eval_shading_normal(scene,
                            scene.instances[intersection.instance],
                            intersection.element, intersection.uv, -incoming),
                        -incoming);
          radiance += weight * bsdfcos * emission / pdf;
        }
        next_emission = false;
      } else {
        next_emission = true;
      }

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (!is_delta(material)) {
        if (rand1f(rng) < 0.5f) {
          incoming = sample_bsdfcos(
              material, normal, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        if (incoming == vec3f{0, 0, 0}) break;
        weight *=
            eval_bsdfcos(material, normal, outgoing, incoming) /
            (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
                0.5f *
                    sample_lights_pdf(scene, bvh, lights, position, incoming));
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        if (incoming == vec3f{0, 0, 0}) break;
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
      }

      // update volume stack
      if (is_volumetric(scene, intersection) &&
          dot(normal, outgoing) * dot(normal, incoming) < 0) {
        if (volume_stack.empty()) {
          auto material = eval_material(scene, intersection);
          volume_stack.push_back(material);
        } else {
          volume_stack.pop_back();
        }
      }

      // setup next iteration
      ray = {position, incoming};
    } else {
      // prepare shading point
      auto  outgoing = -ray.d;
      auto  position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (rand1f(rng) < 0.5f) {
        incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_scattering(vsdf, outgoing, incoming) /
          (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

      // setup next iteration
      ray = {position, incoming};
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing with MIS.
static trace_result trace_pathmis(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto volume_stack  = vector<material_point>{};
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto opbounce      = 0;

  // MIS helpers
  auto mis_heuristic = [](float this_pdf, float other_pdf) {
    return (this_pdf * this_pdf) /
           (this_pdf * this_pdf + other_pdf * other_pdf);
  };
  auto next_emission     = true;
  auto next_intersection = scene_intersection{};

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = next_emission ? intersect_scene(bvh, scene, ray)
                                      : next_intersection;
    if (!intersection.hit) {
      if ((bounce > 0 || !params.envhidden) && next_emission)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // handle transmission if inside a volume
    auto in_volume = false;
    if (!volume_stack.empty()) {
      auto& vsdf     = volume_stack.back();
      auto  distance = sample_transmittance(
          vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
      weight *= eval_transmittance(vsdf.density, distance) /
                sample_transmittance_pdf(
                    vsdf.density, distance, intersection.distance);
      in_volume             = distance < intersection.distance;
      intersection.distance = distance;
    }

    // switch between surface and volume
    if (!in_volume) {
      // prepare shading point
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      // correct roughness
      if (params.nocaustics) {
        max_roughness      = max(material.roughness, max_roughness);
        material.roughness = max_roughness;
      }

      // handle opacity
      if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
        if (opbounce++ > 128) break;
        ray = {position + ray.d * 1e-2f, ray.d};
        bounce -= 1;
        continue;
      }

      // set hit variables
      if (bounce == 0) {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
      }

      // accumulate emission
      if (next_emission) {
        radiance += weight * eval_emission(material, normal, outgoing);
      }

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (!is_delta(material)) {
        // direct with MIS --- light
        for (auto sample_light : {true, false}) {
          incoming = sample_light ? sample_lights(scene, lights, position,
                                        rand1f(rng), rand1f(rng), rand2f(rng))
                                  : sample_bsdfcos(material, normal, outgoing,
                                        rand1f(rng), rand2f(rng));
          if (incoming == vec3f{0, 0, 0}) break;
          auto bsdfcos   = eval_bsdfcos(material, normal, outgoing, incoming);
          auto light_pdf = sample_lights_pdf(
              scene, bvh, lights, position, incoming);
          auto bsdf_pdf = sample_bsdfcos_pdf(
              material, normal, outgoing, incoming);
          auto mis_weight = sample_light
                                ? mis_heuristic(light_pdf, bsdf_pdf) / light_pdf
                                : mis_heuristic(bsdf_pdf, light_pdf) / bsdf_pdf;
          if (bsdfcos != vec3f{0, 0, 0} && mis_weight != 0) {
            auto intersection = intersect_scene(
                bvh, scene, {position, incoming});
            if (!sample_light) next_intersection = intersection;
            auto emission = vec3f{0, 0, 0};
            if (!intersection.hit) {
              emission = eval_environment(scene, incoming);
            } else {
              auto material = eval_material(scene,
                  scene.instances[intersection.instance], intersection.element,
                  intersection.uv);
              emission      = eval_emission(material,
                       eval_shading_normal(scene,
                           scene.instances[intersection.instance],
                           intersection.element, intersection.uv, -incoming),
                       -incoming);
            }
            radiance += weight * bsdfcos * emission * mis_weight;
          }
        }

        // indirect
        weight *= eval_bsdfcos(material, normal, outgoing, incoming) /
                  sample_bsdfcos_pdf(material, normal, outgoing, incoming);
        next_emission = false;
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
        next_emission = true;
      }

      // update volume stack
      if (is_volumetric(scene, intersection) &&
          dot(normal, outgoing) * dot(normal, incoming) < 0) {
        if (volume_stack.empty()) {
          auto material = eval_material(scene, intersection);
          volume_stack.push_back(material);
        } else {
          volume_stack.pop_back();
        }
      }

      // setup next iteration
      ray = {position, incoming};
    } else {
      // prepare shading point
      auto  outgoing = -ray.d;
      auto  position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (rand1f(rng) < 0.5f) {
        incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
        next_emission = true;
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        next_emission = true;
      }
      weight *=
          eval_scattering(vsdf, outgoing, incoming) /
          (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

      // setup next iteration
      ray = {position, incoming};
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing.
static trace_result trace_pathtest(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray_,
    rng_state& rng, const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto opbounce      = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);
    material.type = material_type::matte;

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    radiance += weight * eval_emission(material, normal, outgoing);

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (!is_delta(material)) {
      if (rand1f(rng) < 0.5f) {
        incoming = sample_bsdfcos(
            material, normal, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_bsdfcos(material, normal, outgoing, incoming) /
          (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // setup next iteration
    ray = {position, incoming};

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Recursive path tracing.
static trace_result trace_naive(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    radiance += weight * eval_emission(material, normal, outgoing);

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (material.roughness != 0) {
      incoming = sample_bsdfcos(
          material, normal, outgoing, rand1f(rng), rand2f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_bsdfcos(material, normal, outgoing, incoming) /
                sample_bsdfcos_pdf(material, normal, outgoing, incoming);
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Volume Path tracing NSPI
static trace_result trace_path_volume(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray_,
    rng_state& rng, const trace_params& params) {
  // initialize
  auto radiance      = vec3f{0, 0, 0};
  auto weight        = vec3f{1, 1, 1};
  auto ray           = ray_;
  auto volume_stack  = vector<material_point>{};
  auto max_roughness = 0.0f;
  auto hit           = false;
  auto hit_albedo    = vec3f{0, 0, 0};
  auto hit_normal    = vec3f{0, 0, 0};
  auto opbounce      = 0;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // handle transmission if inside a volume
    auto in_volume = false;
    auto position  = vec3f{0, 0, 0};
    auto incoming  = ray.d;
    auto outgoing  = -ray.d;

    if (!volume_stack.empty()) {
      auto& vsdf = volume_stack.back();
      // heterogeneus volumes NSPI
      if (vsdf.htvolume) {
        // TO DO: implement eval_unidirectional_spectral_mis
        auto [t, w] = eval_unidirectional_spectral_mis_NSPI(
            vsdf, intersection.distance, rng, ray);
        weight *= w;
        position = ray.o + t * ray.d;
        // Handle an interaction with a medium
        if (t < intersection.distance) {
          in_volume = true;
          if (vsdf.event == material_event::scatter)
            incoming = sample_scattering(
                vsdf, outgoing, rand1f(rng), rand2f(rng));
          if (vsdf.event == material_event::absorb) {
            auto er = zero3f;
            /*  Check about emission
            if (has_emission(vsdf)) {
              er = blackbody_to_rgb(eval_vpt_emission(vsdf, position) * 40e3);
            */
            radiance += weight * er * vsdf.volume.radiance_mult;
            break;
          }
        }
      }
      // homogeneus volumes
      else {
        auto distance = sample_transmittance(
            vsdf.density, intersection.distance, rand1f(rng), rand1f(rng));
        weight *= eval_transmittance(vsdf.density, distance) /
                  sample_transmittance_pdf(
                      vsdf.density, distance, intersection.distance);
        in_volume             = distance < intersection.distance;
        intersection.distance = distance;
      }
    }

    // switch between surface and volume
    if (!in_volume) {
      // prepare shading point
      auto outgoing = -ray.d;
      auto position = eval_shading_position(scene, intersection, outgoing);
      auto normal   = eval_shading_normal(scene, intersection, outgoing);
      auto material = eval_material(scene, intersection);

      // correct roughness
      if (params.nocaustics) {
        max_roughness      = max(material.roughness, max_roughness);
        material.roughness = max_roughness;
      }

      // handle opacity
      if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
        if (opbounce++ > 128) break;
        ray = {position + ray.d * 1e-2f, ray.d};
        bounce -= 1;
        continue;
      }

      // set hit variables
      if (bounce == 0) {
        hit        = true;
        hit_albedo = material.color;
        hit_normal = normal;
      }

      // accumulate emission
      radiance += weight * eval_emission(material, normal, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (!is_delta(material)) {
        if (rand1f(rng) < 0.5f) {
          incoming = sample_bsdfcos(
              material, normal, outgoing, rand1f(rng), rand2f(rng));
        } else {
          incoming = sample_lights(
              scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
        }
        if (incoming == vec3f{0, 0, 0}) break;
        weight *=
            eval_bsdfcos(material, normal, outgoing, incoming) /
            (0.5f * sample_bsdfcos_pdf(material, normal, outgoing, incoming) +
                0.5f *
                    sample_lights_pdf(scene, bvh, lights, position, incoming));
      } else {
        incoming = sample_delta(material, normal, outgoing, rand1f(rng));
        weight *= eval_delta(material, normal, outgoing, incoming) /
                  sample_delta_pdf(material, normal, outgoing, incoming);
      }

      // update volume stack
      if (is_volumetric(scene, intersection) &&
          dot(normal, outgoing) * dot(normal, incoming) < 0) {
        if (volume_stack.empty()) {
          auto material = eval_material(scene, intersection);
          volume_stack.push_back(material);
        } else {
          volume_stack.pop_back();
        }
      }

      // setup next iteration
      ray = {position, incoming};
    } else {
      // prepare shading point
      auto  outgoing = -ray.d;
      auto  position = ray.o + ray.d * intersection.distance;
      auto& vsdf     = volume_stack.back();

      // accumulate emission
      // radiance += weight * eval_volemission(emission, outgoing);

      // next direction
      auto incoming = vec3f{0, 0, 0};
      if (rand1f(rng) < 0.5f) {
        incoming = sample_scattering(vsdf, outgoing, rand1f(rng), rand2f(rng));
      } else {
        incoming = sample_lights(
            scene, lights, position, rand1f(rng), rand1f(rng), rand2f(rng));
      }
      if (incoming == vec3f{0, 0, 0}) break;
      weight *=
          eval_scattering(vsdf, outgoing, incoming) /
          (0.5f * sample_scattering_pdf(vsdf, outgoing, incoming) +
              0.5f * sample_lights_pdf(scene, bvh, lights, position, incoming));

      // setup next iteration
      ray = {position, incoming};
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }
  }

  return {radiance, hit, hit_albedo, hit_normal};
}


// Eyelight for quick previewing.
static trace_result trace_eyelight(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray_,
    rng_state& rng, const trace_params& params) {
  // initialize
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  // trace  path
  for (auto bounce = 0; bounce < max(params.bounces, 4); bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    auto incoming = outgoing;
    radiance += weight * eval_emission(material, normal, outgoing);

    // brdf * light
    radiance += weight * pif *
                eval_bsdfcos(material, normal, outgoing, incoming);

    // continue path
    if (!is_delta(material)) break;
    incoming = sample_delta(material, normal, outgoing, rand1f(rng));
    if (incoming == vec3f{0, 0, 0}) break;
    weight *= eval_delta(material, normal, outgoing, incoming) /
              sample_delta_pdf(material, normal, outgoing, incoming);
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Diagram previewing.
static trace_result trace_diagram(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;

  // trace  path
  for (auto bounce = 0; bounce < max(params.bounces, 4); bounce++) {
    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      radiance += weight * vec3f{1, 1, 1};
      hit = true;
      // if (bounce > 0 || !params.envhidden)
      //   radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto outgoing = -ray.d;
    auto position = eval_shading_position(scene, intersection, outgoing);
    auto normal   = eval_shading_normal(scene, intersection, outgoing);
    auto material = eval_material(scene, intersection);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    auto incoming = outgoing;
    radiance += weight * eval_emission(material, normal, outgoing);

    // brdf * light
    radiance += weight * pif *
                eval_bsdfcos(material, normal, outgoing, incoming);

    // continue path
    if (!is_delta(material)) break;
    incoming = sample_delta(material, normal, outgoing, rand1f(rng));
    if (incoming == vec3f{0, 0, 0}) break;
    weight *= eval_delta(material, normal, outgoing, incoming) /
              sample_delta_pdf(material, normal, outgoing, incoming);
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // setup next iteration
    ray = {position, incoming};
  }

  return {radiance, hit, hit_albedo, hit_normal};
}

// Furnace test.
static trace_result trace_furnace(const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const ray3f& ray_, rng_state& rng,
    const trace_params& params) {
  // initialize
  auto radiance   = vec3f{0, 0, 0};
  auto weight     = vec3f{1, 1, 1};
  auto ray        = ray_;
  auto hit        = false;
  auto hit_albedo = vec3f{0, 0, 0};
  auto hit_normal = vec3f{0, 0, 0};
  auto opbounce   = 0;
  auto in_volume  = false;

  // trace  path
  for (auto bounce = 0; bounce < params.bounces; bounce++) {
    // exit loop
    if (bounce > 0 && !in_volume) {
      radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // intersect next point
    auto intersection = intersect_scene(bvh, scene, ray);
    if (!intersection.hit) {
      if (bounce > 0 || !params.envhidden)
        radiance += weight * eval_environment(scene, ray.d);
      break;
    }

    // prepare shading point
    auto  outgoing = -ray.d;
    auto& instance = scene.instances[intersection.instance];
    auto  element  = intersection.element;
    auto  uv       = intersection.uv;
    auto  position = eval_position(scene, instance, element, uv);
    auto  normal = eval_shading_normal(scene, instance, element, uv, outgoing);
    auto  material = eval_material(scene, instance, element, uv);

    // handle opacity
    if (material.opacity < 1 && rand1f(rng) >= material.opacity) {
      if (opbounce++ > 128) break;
      ray = {position + ray.d * 1e-2f, ray.d};
      bounce -= 1;
      continue;
    }

    // set hit variables
    if (bounce == 0) {
      hit        = true;
      hit_albedo = material.color;
      hit_normal = normal;
    }

    // accumulate emission
    radiance += weight * eval_emission(material, normal, outgoing);

    // next direction
    auto incoming = vec3f{0, 0, 0};
    if (material.roughness != 0) {
      incoming = sample_bsdfcos(
          material, normal, outgoing, rand1f(rng), rand2f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_bsdfcos(material, normal, outgoing, incoming) /
                sample_bsdfcos_pdf(material, normal, outgoing, incoming);
    } else {
      incoming = sample_delta(material, normal, outgoing, rand1f(rng));
      if (incoming == vec3f{0, 0, 0}) break;
      weight *= eval_delta(material, normal, outgoing, incoming) /
                sample_delta_pdf(material, normal, outgoing, incoming);
    }

    // check weight
    if (weight == vec3f{0, 0, 0} || !isfinite(weight)) break;

    // russian roulette
    if (bounce > 3) {
      auto rr_prob = min((float)0.99, max(weight));
      if (rand1f(rng) >= rr_prob) break;
      weight *= 1 / rr_prob;
    }

    // update volume stack
    if (dot(normal, outgoing) * dot(normal, incoming) < 0)
      in_volume = !in_volume;

    // setup next iteration
    ray = {position, incoming};
  }

  // done
  return {radiance, hit, hit_albedo, hit_normal};
}

// False color rendering
static trace_result trace_falsecolor(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray,
    rng_state& rng, const trace_params& params) {
  // intersect next point
  auto intersection = intersect_scene(bvh, scene, ray);
  if (!intersection.hit) return {};

  // prepare shading point
  auto outgoing = -ray.d;
  auto position = eval_shading_position(scene, intersection, outgoing);
  auto normal   = eval_shading_normal(scene, intersection, outgoing);
  auto gnormal  = eval_element_normal(scene, intersection);
  auto texcoord = eval_texcoord(scene, intersection);
  auto material = eval_material(scene, intersection);
  auto delta    = is_delta(material) ? 1.0f : 0.0f;

  // hash color
  auto hashed_color = [](int id) {
    auto hashed = std::hash<int>()(id);
    auto rng    = make_rng(trace_default_seed, hashed);
    return pow(0.5f + 0.5f * rand3f(rng), 2.2f);
  };

  // compute result
  auto result = vec3f{0, 0, 0};
  switch (params.falsecolor) {
    case trace_falsecolor_type::position:
      result = position * 0.5f + 0.5f;
      break;
    case trace_falsecolor_type::normal: result = normal * 0.5f + 0.5f; break;
    case trace_falsecolor_type::frontfacing:
      result = dot(normal, -ray.d) > 0 ? vec3f{0, 1, 0} : vec3f{1, 0, 0};
      break;
    case trace_falsecolor_type::gnormal: result = gnormal * 0.5f + 0.5f; break;
    case trace_falsecolor_type::gfrontfacing:
      result = dot(gnormal, -ray.d) > 0 ? vec3f{0, 1, 0} : vec3f{1, 0, 0};
      break;
    case trace_falsecolor_type::mtype:
      result = hashed_color((int)material.type);
      break;
    case trace_falsecolor_type::texcoord:
      result = {fmod(texcoord.x, 1.0f), fmod(texcoord.y, 1.0f), 0};
      break;
    case trace_falsecolor_type::color: result = material.color; break;
    case trace_falsecolor_type::emission: result = material.emission; break;
    case trace_falsecolor_type::roughness:
      result = {material.roughness, material.roughness, material.roughness};
      break;
    case trace_falsecolor_type::opacity:
      result = {material.opacity, material.opacity, material.opacity};
      break;
    case trace_falsecolor_type::metallic:
      result = {material.metallic, material.metallic, material.metallic};
      break;
    case trace_falsecolor_type::delta: result = {delta, delta, delta}; break;
    case trace_falsecolor_type::element:
      result = hashed_color(intersection.element);
      break;
    case trace_falsecolor_type::instance:
      result = hashed_color(intersection.instance);
      break;
    case trace_falsecolor_type::shape:
      result = hashed_color(scene.instances[intersection.instance].shape);
      break;
    case trace_falsecolor_type::material:
      result = hashed_color(scene.instances[intersection.instance].material);
      break;
    case trace_falsecolor_type::highlight: {
      if (material.emission == vec3f{0, 0, 0})
        material.emission = {0.2f, 0.2f, 0.2f};
      result = material.emission * abs(dot(-ray.d, normal));
      break;
    } break;
    default: result = {0, 0, 0};
  }

  // done
  return {srgb_to_rgb(result), true, material.color, normal};
}

// Trace a single ray from the camera using the given algorithm.
using sampler_func = trace_result (*)(const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, const ray3f& ray,
    rng_state& rng, const trace_params& params);
static sampler_func get_trace_sampler_func(const trace_params& params) {
  switch (params.sampler) {
    case trace_sampler_type::path: return trace_path;
    case trace_sampler_type::pathdirect: return trace_pathdirect;
    case trace_sampler_type::pathmis: return trace_pathmis;
    case trace_sampler_type::pathtest: return trace_pathtest;
    case trace_sampler_type::naive: return trace_naive;
    case trace_sampler_type::eyelight: return trace_eyelight;
    case trace_sampler_type::diagram: return trace_diagram;
    case trace_sampler_type::furnace: return trace_furnace;
    case trace_sampler_type::falsecolor: return trace_falsecolor;
    default: {
      throw std::runtime_error("sampler unknown");
      return nullptr;
    }
  }
}

// Check is a sampler requires lights
bool is_sampler_lit(const trace_params& params) {
  switch (params.sampler) {
    case trace_sampler_type::path: return true;
    case trace_sampler_type::pathdirect: return true;
    case trace_sampler_type::pathmis: return true;
    case trace_sampler_type::naive: return true;
    case trace_sampler_type::eyelight: return false;
    case trace_sampler_type::furnace: return true;
    case trace_sampler_type::falsecolor: return false;
    default: {
      throw std::runtime_error("sampler unknown");
      return false;
    }
  }
}

// Trace a block of samples
void trace_sample(trace_state& state, const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights, int i, int j, int sample,
    const trace_params& params) {
  auto& camera  = scene.cameras[params.camera];
  auto  sampler = get_trace_sampler_func(params);
  auto  idx     = state.width * j + i;
  auto  ray     = sample_camera(camera, {i, j}, {state.width, state.height},
           rand2f(state.rngs[idx]), rand2f(state.rngs[idx]), params.tentfilter);
  auto [radiance, hit, albedo, normal] = sampler(
      scene, bvh, lights, ray, state.rngs[idx], params);
  if (!isfinite(radiance)) radiance = {0, 0, 0};
  if (max(radiance) > params.clamp)
    radiance = radiance * (params.clamp / max(radiance));
  auto weight = 1.0f / (sample + 1);
  if (hit) {
    state.image[idx] = lerp(
        state.image[idx], {radiance.x, radiance.y, radiance.z, 1}, weight);
    state.albedo[idx] = lerp(state.albedo[idx], albedo, weight);
    state.normal[idx] = lerp(state.normal[idx], normal, weight);
    state.hits[idx] += 1;
  } else if (!params.envhidden && !scene.environments.empty()) {
    state.image[idx] = lerp(
        state.image[idx], {radiance.x, radiance.y, radiance.z, 1}, weight);
    state.albedo[idx] = lerp(state.albedo[idx], {1, 1, 1}, weight);
    state.normal[idx] = lerp(state.normal[idx], -ray.d, weight);
    state.hits[idx] += 1;
  } else {
    state.image[idx]  = lerp(state.image[idx], {0, 0, 0, 0}, weight);
    state.albedo[idx] = lerp(state.albedo[idx], {0, 0, 0}, weight);
    state.normal[idx] = lerp(state.normal[idx], -ray.d, weight);
  }
}

// Init a sequence of random number generators.
trace_state make_trace_state(
    const scene_data& scene, const trace_params& params) {
  auto& camera = scene.cameras[params.camera];
  auto  state  = trace_state{};
  if (camera.aspect >= 1) {
    state.width  = params.resolution;
    state.height = (int)round(params.resolution / camera.aspect);
  } else {
    state.height = params.resolution;
    state.width  = (int)round(params.resolution * camera.aspect);
  }
  state.samples = 0;
  state.image.assign(state.width * state.height, {0, 0, 0, 0});
  state.albedo.assign(state.width * state.height, {0, 0, 0});
  state.normal.assign(state.width * state.height, {0, 0, 0});
  state.hits.assign(state.width * state.height, 0);
  state.rngs.assign(state.width * state.height, {});
  auto rng_ = make_rng(1301081);
  for (auto& rng : state.rngs) {
    rng = make_rng(params.seed, rand1i(rng_, 1 << 31) / 2 + 1);
  }
  if (params.denoise) {
    state.denoised.assign(state.width * state.height, {0, 0, 0, 0});
  }
  return state;
}

// Forward declaration
static trace_light& add_light(trace_lights& lights) {
  return lights.lights.emplace_back();
}

// Init trace lights
trace_lights make_trace_lights(
    const scene_data& scene, const trace_params& params) {
  auto lights = trace_lights{};

  for (auto handle : range(scene.instances.size())) {
    auto& instance = scene.instances[handle];
    auto& material = scene.materials[instance.material];
    if (material.emission == vec3f{0, 0, 0}) continue;
    auto& shape = scene.shapes[instance.shape];
    if (shape.triangles.empty() && shape.quads.empty()) continue;
    auto& light       = add_light(lights);
    light.instance    = (int)handle;
    light.environment = invalidid;
    if (!shape.triangles.empty()) {
      light.elements_cdf = vector<float>(shape.triangles.size());
      for (auto idx : range(light.elements_cdf.size())) {
        auto& t                 = shape.triangles[idx];
        light.elements_cdf[idx] = triangle_area(
            shape.positions[t.x], shape.positions[t.y], shape.positions[t.z]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
    if (!shape.quads.empty()) {
      light.elements_cdf = vector<float>(shape.quads.size());
      for (auto idx : range(light.elements_cdf.size())) {
        auto& t                 = shape.quads[idx];
        light.elements_cdf[idx] = quad_area(shape.positions[t.x],
            shape.positions[t.y], shape.positions[t.z], shape.positions[t.w]);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }
  for (auto handle : range(scene.environments.size())) {
    auto& environment = scene.environments[handle];
    if (environment.emission == vec3f{0, 0, 0}) continue;
    auto& light       = add_light(lights);
    light.instance    = invalidid;
    light.environment = (int)handle;
    if (environment.emission_tex != invalidid) {
      auto& texture      = scene.textures[environment.emission_tex];
      light.elements_cdf = vector<float>(texture.width * texture.height);
      for (auto idx : range(light.elements_cdf.size())) {
        auto ij    = vec2i{(int)idx % texture.width, (int)idx / texture.width};
        auto th    = (ij.y + 0.5f) * pif / texture.height;
        auto value = lookup_texture(texture, ij.x, ij.y);
        light.elements_cdf[idx] = max(value) * sin(th);
        if (idx != 0) light.elements_cdf[idx] += light.elements_cdf[idx - 1];
      }
    }
  }

  // handle progress
  return lights;
}

// Progressively computes an image.
image_data trace_image(const scene_data& scene, const trace_params& params) {
  auto bvh    = make_trace_bvh(scene, params);
  auto lights = make_trace_lights(scene, params);
  auto state  = make_trace_state(scene, params);
  for (auto sample = 0; sample < params.samples; sample++) {
    trace_samples(state, scene, bvh, lights, params);
  }
  return get_image(state);
}

// Progressively compute an image by calling trace_samples multiple times.
void trace_samples(trace_state& state, const scene_data& scene,
    const trace_bvh& bvh, const trace_lights& lights,
    const trace_params& params) {
  if (state.samples >= params.samples) return;
  if (params.noparallel) {
    for (auto j : range(state.height)) {
      for (auto i : range(state.width)) {
        for (auto sample : range(state.samples, state.samples + params.batch)) {
          trace_sample(state, scene, bvh, lights, i, j, sample, params);
        }
      }
    }
  } else {
    parallel_for(state.width, state.height, [&](int i, int j) {
      for (auto sample : range(state.samples, state.samples + params.batch)) {
        trace_sample(state, scene, bvh, lights, i, j, sample, params);
      }
    });
  }
  state.samples += params.batch;
  if (params.denoise && !state.denoised.empty()) {
    denoise_image(state.denoised, state.width, state.height, state.image,
        state.albedo, state.normal);
  }
}

// Trace context
trace_context make_trace_context(const trace_params& params) {
  return {{}, false, false};
}

// Async start
void trace_start(trace_context& context, trace_state& state,
    const scene_data& scene, const trace_bvh& bvh, const trace_lights& lights,
    const trace_params& params) {
  if (state.samples >= params.samples) return;
  context.stop   = false;
  context.done   = false;
  context.worker = std::async(std::launch::async, [&]() {
    if (context.stop) return;
    parallel_for(state.width, state.height, [&](int i, int j) {
      for (auto sample : range(state.samples, state.samples + params.batch)) {
        if (context.stop) return;
        trace_sample(state, scene, bvh, lights, i, j, sample, params);
      }
    });
    state.samples += params.batch;
    if (context.stop) return;
    if (params.denoise && !state.denoised.empty()) {
      denoise_image(state.denoised, state.width, state.height, state.image,
          state.albedo, state.normal);
    }
    context.done = true;
  });
}

// Async cancel
void trace_cancel(trace_context& context) {
  context.stop = true;
  if (context.worker.valid()) context.worker.get();
}

// Async done
bool trace_done(const trace_context& context) { return context.done; }

void trace_preview(color_image& image, trace_context& context,
    trace_state& state, const scene_data& scene, const trace_bvh& bvh,
    const trace_lights& lights, const trace_params& params) {
  // preview
  auto pparams = params;
  pparams.resolution /= params.pratio;
  pparams.samples = 1;
  auto pstate     = make_trace_state(scene, pparams);
  trace_samples(pstate, scene, bvh, lights, pparams);
  auto preview = get_image(pstate);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    auto i = idx % image.width, j = idx / image.width;
    auto pi           = clamp(i / params.pratio, 0, preview.width - 1),
         pj           = clamp(j / params.pratio, 0, preview.height - 1);
    image.pixels[idx] = preview.pixels[pj * preview.width + pi];
  }
};

// Check image type
static void check_image(
    const image_data& image, int width, int height, bool linear) {
  if (image.width != width || image.height != height)
    throw std::invalid_argument{"image should have the same size"};
  if (image.linear != linear)
    throw std::invalid_argument{
        linear ? "expected linear image" : "expected srgb image"};
}
template <typename T>
static void check_image(const vector<T>& image, int width, int height) {
  if (image.size() != (size_t)width * (size_t)height)
    throw std::invalid_argument{"image should have the same size"};
}

// Get resulting render, denoised if requested
image_data get_image(const trace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_image(image, state);
  return image;
}
void get_image(image_data& image, const trace_state& state) {
  image.width  = state.width;
  image.height = state.height;
  image.linear = true;
  if (state.denoised.empty()) {
    image.pixels = state.image;
  } else {
    image.pixels = state.denoised;
  }
}

// Get resulting render
image_data get_rendered_image(const trace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_rendered_image(image, state);
  return image;
}
void get_rendered_image(image_data& image, const trace_state& state) {
  check_image(image, state.width, state.height, true);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    image.pixels[idx] = state.image[idx];
  }
}

// Get denoised render
image_data get_denoised_image(const trace_state& state) {
  auto image = make_image(state.width, state.height, true);
  get_denoised_image(image, state);
  return image;
}
void get_denoised_image(image_data& image, const trace_state& state) {
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // get image
  get_rendered_image(image, state);

  // get albedo and normal
  auto albedo = vector<vec3f>(image.pixels.size()),
       normal = vector<vec3f>(image.pixels.size());
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    albedo[idx] = state.albedo[idx];
    normal[idx] = state.normal[idx];
  }

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.setImage("albedo", (void*)albedo.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("normal", (void*)normal.data(), oidn::Format::Float3,
      state.width, state.height);
  filter.setImage("output", image.pixels.data(), oidn::Format::Float3,
      state.width, state.height, 0, sizeof(vec4f), sizeof(vec4f) * state.width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  get_rendered_image(image, state);
#endif
}

// Get denoising buffers
image_data get_albedo_image(const trace_state& state) {
  auto albedo = make_image(state.width, state.height, true);
  get_albedo_image(albedo, state);
  return albedo;
}
void get_albedo_image(image_data& albedo, const trace_state& state) {
  check_image(albedo, state.width, state.height, true);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    albedo.pixels[idx] = {
        state.albedo[idx].x, state.albedo[idx].y, state.albedo[idx].z, 1.0f};
  }
}
image_data get_normal_image(const trace_state& state) {
  auto normal = make_image(state.width, state.height, true);
  get_normal_image(normal, state);
  return normal;
}
void get_normal_image(image_data& normal, const trace_state& state) {
  check_image(normal, state.width, state.height, true);
  for (auto idx = 0; idx < state.width * state.height; idx++) {
    normal.pixels[idx] = {
        state.normal[idx].x, state.normal[idx].y, state.normal[idx].z, 1.0f};
  }
}

// Denoise image
image_data denoise_image(const image_data& render, const image_data& albedo,
    const image_data& normal) {
  auto denoised = make_image(render.width, render.height, render.linear);
  denoise_image(denoised, render, albedo, normal);
  return denoised;
}
void denoise_image(image_data& denoised, const image_data& render,
    const image_data& albedo, const image_data& normal) {
  check_image(denoised, render.width, render.height, render.linear);
  check_image(albedo, render.width, render.height, albedo.linear);
  check_image(normal, render.width, render.height, normal.linear);
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // set image
  denoised = render;

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)render.pixels.data(), oidn::Format::Float3,
      render.width, render.height, 0, sizeof(vec4f),
      sizeof(vec4f) * render.width);
  filter.setImage("albedo", (void*)albedo.pixels.data(), oidn::Format::Float3,
      albedo.width, albedo.height, 0, sizeof(vec4f),
      sizeof(vec4f) * albedo.width);
  filter.setImage("normal", (void*)normal.pixels.data(), oidn::Format::Float3,
      normal.width, normal.height, 0, sizeof(vec4f),
      sizeof(vec4f) * normal.width);
  filter.setImage("output", denoised.pixels.data(), oidn::Format::Float3,
      denoised.width, denoised.height, 0, sizeof(vec4f),
      sizeof(vec4f) * denoised.width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  denoised = render;
#endif
}

void denoise_image(vector<vec4f>& denoised, int width, int height,
    const vector<vec4f>& render, const vector<vec3f>& albedo,
    const vector<vec3f>& normal) {
  check_image(denoised, width, height);
  check_image(render, width, height);
  check_image(albedo, width, height);
  check_image(normal, width, height);
#if YOCTO_DENOISE
  // Create an Intel Open Image Denoise device
  oidn::DeviceRef device = oidn::newDevice();
  device.commit();

  // set image
  denoised = render;

  // Create a denoising filter
  oidn::FilterRef filter = device.newFilter("RT");  // ray tracing filter
  filter.setImage("color", (void*)render.data(), oidn::Format::Float3, width,
      height, 0, sizeof(vec4f), sizeof(vec4f) * width);
  filter.setImage("albedo", (void*)albedo.data(), oidn::Format::Float3, width,
      height, 0, sizeof(vec3f), sizeof(vec3f) * width);
  filter.setImage("normal", (void*)normal.data(), oidn::Format::Float3, width,
      height, 0, sizeof(vec3f), sizeof(vec3f) * width);
  filter.setImage("output", denoised.data(), oidn::Format::Float3, width,
      height, 0, sizeof(vec4f), sizeof(vec4f) * width);
  filter.set("inputScale", 1.0f);  // set scale as fixed
  filter.set("hdr", true);         // image is HDR
  filter.commit();

  // Filter the image
  filter.execute();
#else
  denoised = render;
#endif
}

}  // namespace yocto
