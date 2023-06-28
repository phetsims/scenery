// Copyright 2023, University of Colorado Boulder

// Shaders and info have the following license:

/*
Copyright (c) 2020 Raph Levien

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
 */


/**
 * Shader data for Vello.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { DeviceContext, scenery, WorkgroupSize } from '../../imports.js';
import fine from './shaders/fine.js';
import backdrop_dyn from './shaders/backdrop_dyn.js';
import bbox_clear from './shaders/bbox_clear.js';
import binning from './shaders/binning.js';
import clip_leaf from './shaders/clip_leaf.js';
import clip_reduce from './shaders/clip_reduce.js';
import coarse from './shaders/coarse.js';
import draw_leaf from './shaders/draw_leaf.js';
import draw_reduce from './shaders/draw_reduce.js';
import path_coarse from './shaders/path_coarse.js';
import path_coarse_full from './shaders/path_coarse_full.js';
import pathseg from './shaders/pathseg.js';
import pathtag_reduce from './shaders/pathtag_reduce.js';
import pathtag_reduce2 from './shaders/pathtag_reduce2.js';
import pathtag_scan1 from './shaders/pathtag_scan1.js';
import pathtag_scan_large from './shaders/pathtag_scan_large.js';
import pathtag_scan_small from './shaders/pathtag_scan_small.js';
import tile_alloc from './shaders/tile_alloc.js';

type Binding = 'Uniform' | 'BufReadOnly' | 'Buffer' | 'Image' | 'ImageRead';

type ShaderOptions = {
  wgsl: string;
  bindings: Binding[];
};

type VelloShaderFormat = 'rgba8unorm' | 'bgra8unorm';

const BUFFER_TYPE_MAP = {
  Buffer: 'storage',
  BufReadOnly: 'read-only-storage',
  Uniform: 'uniform'
} as const;

export type ShaderMap = {
  backdrop_dyn: VelloShader;
  bbox_clear: VelloShader;
  binning: VelloShader;
  clip_leaf: VelloShader;
  clip_reduce: VelloShader;
  coarse: VelloShader;
  draw_leaf: VelloShader;
  draw_reduce: VelloShader;
  fine_rgba8unorm: VelloShader;
  fine_bgra8unorm: VelloShader;
  path_coarse: VelloShader;
  path_coarse_full: VelloShader;
  pathseg: VelloShader;
  pathtag_reduce: VelloShader;
  pathtag_reduce2: VelloShader;
  pathtag_scan1: VelloShader;
  pathtag_scan_large: VelloShader;
  pathtag_scan_small: VelloShader;
  tile_alloc: VelloShader;
};

// device => shader map
const shaderDeviceMap = new WeakMap<GPUDevice, ShaderMap>();

export default class VelloShader {

  public readonly wgsl!: string;
  public readonly bindings!: Binding[];

  public readonly module!: GPUShaderModule;
  public readonly bindGroupLayout!: GPUBindGroupLayout;
  public readonly pipeline!: GPUComputePipeline;

  public constructor(
    public readonly name: string,
    data: ShaderOptions,
    public readonly device: GPUDevice,
    format: VelloShaderFormat = 'rgba8unorm'
  ) {

    if ( data.wgsl.length === 0 ) {
      // Just bail out, don't try compiling a shader that won't work
      return;
    }

    this.wgsl = data.wgsl;
    this.bindings = data.bindings;

    this.module = device.createShaderModule( {
      label: name,
      code: this.wgsl
    } );

    this.bindGroupLayout = device.createBindGroupLayout( {
      label: `${name} bindGroupLayout`,
      entries: this.bindings.map( ( binding, i ) => {
        const entry: GPUBindGroupLayoutEntry = {
          binding: i,
          visibility: GPUShaderStage.COMPUTE
        };

        if ( binding === 'Buffer' || binding === 'BufReadOnly' || binding === 'Uniform' ) {
          entry.buffer = {
            type: BUFFER_TYPE_MAP[ binding ],
            hasDynamicOffset: false
          };
        }
        else if ( binding === 'Image' ) {
          entry.storageTexture = {
            access: 'write-only',
            format: format,
            viewDimension: '2d'
          };
        }
        else if ( binding === 'ImageRead' ) {
          // Note: fine takes ImageFormat::Rgba8 for Image/ImageRead
          entry.texture = {
            sampleType: 'float',
            viewDimension: '2d',
            multisampled: false
          };
        }
        else {
          throw new Error( `unknown binding: ${binding}` );
        }

        return entry;
      } )
    } );

    this.pipeline = device.createComputePipeline( {
      label: `${name} pipeline`,
      layout: device.createPipelineLayout( {
        bindGroupLayouts: [ this.bindGroupLayout ]
      } ),
      compute: {
        module: this.module,
        entryPoint: 'main'
      }
    } );
  }

  public dispatch( encoder: GPUCommandEncoder, workgroupSize: WorkgroupSize, resources: ( GPUBuffer | GPUTextureView )[] ): void {
    const bindGroup = this.device.createBindGroup( {
      label: `${this.name} bindGroup`,
      layout: this.bindGroupLayout,
      entries: VelloShader.resourcesToEntries( resources )
    } );
    const computePass = encoder.beginComputePass( {
      label: `${this.name} compute pass`
    } );
    computePass.setPipeline( this.pipeline );
    computePass.setBindGroup( 0, bindGroup );
    computePass.dispatchWorkgroups( workgroupSize.x, workgroupSize.y, workgroupSize.z );
    computePass.end();
  }

  private static resourcesToEntries( resources: ( GPUBuffer | GPUTextureView )[] ): GPUBindGroupEntry[] {
    return resources.map( ( resources, i ) => ( {
      binding: i,
      // handle GPUTextureView
      resource: resources instanceof GPUBuffer ? { buffer: resources } : resources
    } ) );
  }

  public static getShaders( device: GPUDevice ): ShaderMap {
    if ( !shaderDeviceMap.has( device ) ) {
      shaderDeviceMap.set( device, VelloShader.loadShaders( device ) );
    }

    const map = shaderDeviceMap.get( device );

    assert && assert( map, 'Should have a map!' );
    return map!;
  }

  private static loadShaders( device: GPUDevice ): ShaderMap {
    const getFineShaderWGSL = ( format: VelloShaderFormat ) => {
      if ( !DeviceContext.supportsBGRATextureStorage && format === 'bgra8unorm' ) {
        // NO shader! Don't compile it
        return '';
      }

      return fine.replace( 'rgba8unorm', format );
    };

    return {
      backdrop_dyn: new VelloShader( 'backdrop_dyn', {
        wgsl: backdrop_dyn,
        bindings: [ 'Uniform', 'BufReadOnly', 'Buffer' ]
      }, device ),
      bbox_clear: new VelloShader( 'bbox_clear', {
        wgsl: bbox_clear,
        bindings: [ 'Uniform', 'Buffer' ]
      }, device ),
      binning: new VelloShader( 'binning', {
        wgsl: binning,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer', 'Buffer', 'Buffer' ]
      }, device ),
      clip_leaf: new VelloShader( 'clip_leaf', {
        wgsl: clip_leaf,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer' ]
      }, device ),
      clip_reduce: new VelloShader( 'clip_reduce', {
        wgsl: clip_reduce,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer' ]
      }, device ),
      coarse: new VelloShader( 'coarse', {
        wgsl: coarse,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer' ]
      }, device ),
      draw_leaf: new VelloShader( 'draw_leaf', {
        wgsl: draw_leaf,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer', 'Buffer' ]
      }, device ),
      draw_reduce: new VelloShader( 'draw_reduce', {
        wgsl: draw_reduce,
        bindings: [ 'Uniform', 'BufReadOnly', 'Buffer' ]
      }, device ),
      fine_rgba8unorm: new VelloShader( 'fine_rgba8unorm', {
        wgsl: getFineShaderWGSL( 'rgba8unorm' ),
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Image', 'BufReadOnly', 'ImageRead', 'BufReadOnly', 'ImageRead' ]
      }, device, 'rgba8unorm' ),
      fine_bgra8unorm: new VelloShader( 'fine_bgra8unorm', {
        wgsl: getFineShaderWGSL( 'bgra8unorm' ),
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Image', 'BufReadOnly', 'ImageRead', 'BufReadOnly', 'ImageRead' ]
      }, device, 'bgra8unorm' ),
      path_coarse: new VelloShader( 'path_coarse', {
        wgsl: path_coarse,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer' ]
      }, device ),
      path_coarse_full: new VelloShader( 'path_coarse_full', {
        wgsl: path_coarse_full,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer', 'Buffer' ]
      }, device ),
      pathseg: new VelloShader( 'pathseg', {
        wgsl: pathseg,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer' ]
      }, device ),
      pathtag_reduce: new VelloShader( 'pathtag_reduce', {
        wgsl: pathtag_reduce,
        bindings: [ 'Uniform', 'BufReadOnly', 'Buffer' ]
      }, device ),
      pathtag_reduce2: new VelloShader( 'pathtag_reduce2', {
        wgsl: pathtag_reduce2,
        bindings: [ 'BufReadOnly', 'Buffer' ]
      }, device ),
      pathtag_scan1: new VelloShader( 'pathtag_scan1', {
        wgsl: pathtag_scan1,
        bindings: [ 'BufReadOnly', 'BufReadOnly', 'Buffer' ]
      }, device ),
      pathtag_scan_large: new VelloShader( 'pathtag_scan_large', {
        wgsl: pathtag_scan_large,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer' ]
      }, device ),
      pathtag_scan_small: new VelloShader( 'pathtag_scan_small', {
        wgsl: pathtag_scan_small,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer' ]
      }, device ),
      tile_alloc: new VelloShader( 'tile_alloc', {
        wgsl: tile_alloc,
        bindings: [ 'Uniform', 'BufReadOnly', 'BufReadOnly', 'Buffer', 'Buffer', 'Buffer' ]
      }, device )
    };
  }
}

scenery.register( 'VelloShader', VelloShader );
