// Copyright 2023, University of Colorado Boulder

/**
 * Handles resources related to a GPUDevice
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Atlas, Ramps, scenery, VelloShader } from '../../imports.js';
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';

export default class DeviceContext {

  public readonly ramps: Ramps;
  public readonly atlas: Atlas;
  public readonly preferredCanvasFormat: GPUTextureFormat; // TODO: support other formats?
  public readonly preferredStorageFormat: 'bgra8unorm' | 'rgba8unorm';

  public static currentDevice: GPUDevice | null = null;
  public static currentDeviceContext: DeviceContext | null = null;
  public static supportsBGRATextureStorage = false;
  private static couldNotGetDevice = false;
  private static completedDeviceAttempt = false;

  public blitShaderModule: GPUShaderModule | null = null;
  public blitShaderBindGroupLayout: GPUBindGroupLayout | null = null;
  public blitShaderPipelineLayout: GPUPipelineLayout | null = null;
  public blitShaderPipeline: GPURenderPipeline | null = null;

  public lostEmitter = new TinyEmitter();

  public constructor( public device: GPUDevice ) {
    this.ramps = new Ramps( device );
    this.atlas = new Atlas( device );

    // Trigger shader compilation before anything (will be cached)
    VelloShader.getShaders( device );

    this.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.preferredStorageFormat = this.preferredCanvasFormat === 'bgra8unorm' ? 'bgra8unorm' : 'rgba8unorm';
    if ( this.preferredCanvasFormat === 'bgra8unorm' && !device.features.has( 'bgra8unorm-storage' ) ) {
      // TODO: will we need a texture copy?
      this.preferredStorageFormat = 'rgba8unorm';
    }

    if ( this.preferredCanvasFormat !== this.preferredStorageFormat ) {
      this.blitShaderModule = device.createShaderModule( {
        label: 'blitShaderModule',
        code: `@vertex
fn vs_main(@builtin(vertex_index) ix: u32) -> @builtin(position) vec4<f32> {
    // Generate a full screen quad in NDCs
    var vertex = vec2(-1.0, 1.0);
    switch ix {
        case 1u: {
            vertex = vec2(-1.0, -1.0);
        }
        case 2u, 4u: {
            vertex = vec2(1.0, -1.0);
        }
        case 5u: {
            vertex = vec2(1.0, 1.0);
        }
        default: {}
    }
    return vec4(vertex, 0.0, 1.0);
}

@group(0) @binding(0)
var fine_output: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(fine_output, vec2<i32>(pos.xy), 0);
}
        `
      } );

      this.blitShaderBindGroupLayout = device.createBindGroupLayout( {
        label: 'blitShaderBindGroupLayout',
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {
              sampleType: 'float',
              viewDimension: '2d',
              multisampled: false
            }
          }
        ]
      } );

      this.blitShaderPipelineLayout = device.createPipelineLayout( {
        label: 'blitShaderPipelineLayout',
        bindGroupLayouts: [ this.blitShaderBindGroupLayout ]
      } );

      this.blitShaderPipeline = device.createRenderPipeline( {
        label: 'blitShaderPipeline',
        layout: this.blitShaderPipelineLayout,
        vertex: {
          module: this.blitShaderModule,
          entryPoint: 'vs_main'
        },
        fragment: {
          module: this.blitShaderModule,
          entryPoint: 'fs_main',
          targets: [
            {
              format: this.preferredCanvasFormat
            }
          ]
          // TODO: other things to specify? see vello's lib.rs
        }
      } );
    }

    // TODO: handle context losses, reconstruct with the device
    device.lost.then( info => {
      console.error( `WebGPU device was lost: ${info.message}` );

      this.lostEmitter.emit();

      DeviceContext.currentDevice = null;
      DeviceContext.currentDeviceContext = null;

      // 'reason' will be 'destroyed' if we intentionally destroy the device.
      if ( info.reason !== 'destroyed' ) {
        // TODO: handle destruction
      }
    } ).catch( err => {
      throw new Error( err );
    } );
  }

  public getCanvasContext( canvas: HTMLCanvasElement ): GPUCanvasContext {
    const context = canvas.getContext( 'webgpu' )!;

    if ( !context ) {
      throw new Error( 'Could not get a WebGPU context for the given Canvas' );
    }

    context.configure( {
      device: this.device,
      format: this.preferredCanvasFormat,
      usage: GPUTextureUsage.COPY_SRC |
             GPUTextureUsage.RENDER_ATTACHMENT |
             ( this.preferredCanvasFormat === this.preferredStorageFormat ? GPUTextureUsage.STORAGE_BINDING : 0 ),

      // Very important, otherwise we're opaque by default and alpha is ignored. We need to stack!!!
      alphaMode: 'premultiplied'
    } );
    return context;
  }

  // TODO: have something call this early, so the await is not needed?
  public static async isVelloSupported(): Promise<boolean> {
    return !!await DeviceContext.getDevice();
  }

  // TODO: have a Property perhaps?
  public static isVelloSupportedSync(): boolean {
    assert && assert( DeviceContext.completedDeviceAttempt, 'We should have awaited isVelloSupported() before calling this' );

    return !!DeviceContext.currentDevice;
  }

  public static async getDevice(): Promise<GPUDevice | null> {
    if ( DeviceContext.currentDevice ) {
      return DeviceContext.currentDevice;
    }
    if ( DeviceContext.couldNotGetDevice ) {
      // Don't retry attempts to get a device if one failed
      return null;
    }

    let device: GPUDevice | null = null;

    try {
      const adapter = await navigator.gpu?.requestAdapter( {
        powerPreference: 'high-performance'
      } );

      // console.log( [ ...( adapter?.features || [] ) ] );

      DeviceContext.supportsBGRATextureStorage = adapter?.features.has( 'bgra8unorm-storage' ) || false;

      device = await adapter?.requestDevice( {
        requiredFeatures: DeviceContext.supportsBGRATextureStorage ? [ 'bgra8unorm-storage' ] : []
      } ) || null;
    }
    catch( err ) {
      // For now, do nothing (WebGPU not enabled message perhaps?)
      console.log( err );
    }

    DeviceContext.completedDeviceAttempt = true;

    if ( device ) {
      DeviceContext.currentDevice = device;
    }
    else {
      DeviceContext.couldNotGetDevice = true;
    }

    return device || null;
  }

  // Just try to get a device context synchronously
  public static getSync(): DeviceContext | null {
    assert && assert( DeviceContext.completedDeviceAttempt, 'We should have awaited isVelloSupported() before calling this' );

    if ( DeviceContext.currentDeviceContext ) {
      return DeviceContext.currentDeviceContext;
    }

    if ( DeviceContext.currentDevice ) {
      return new DeviceContext( DeviceContext.currentDevice );
    }
    else {
      return null;
    }
  }

  // TODO: rename
  public static async create(): Promise<DeviceContext | null> {
    if ( DeviceContext.currentDeviceContext ) {
      return DeviceContext.currentDeviceContext;
    }

    const device = await DeviceContext.getDevice();

    return device ? new DeviceContext( device ) : null;
  }

  public dispose(): void {
    this.ramps.dispose();
    this.atlas.dispose();

    this.device.destroy();
  }
}

const velloLock = asyncLoader.createLock( { name: 'vello' } );
( async () => {
  // This will cache it
  await DeviceContext.isVelloSupported();

  velloLock();
} )().catch( err => {
  velloLock();
  console.log( err );
} );

scenery.register( 'DeviceContext', DeviceContext );
