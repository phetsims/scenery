// Copyright 2023, University of Colorado Boulder

/**
 * Handles resources related to a GPUDevice
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Atlas from './Atlas.js';
import Ramps from './Ramps.js';
import { scenery } from '../../imports.js';
import VelloShader from './VelloShader.js';

export default class DeviceContext {

  public ramps: Ramps;
  public atlas: Atlas;
  public preferredCanvasFormat: GPUTextureFormat; // TODO: support other formats?

  public constructor( public device: GPUDevice ) {
    this.ramps = new Ramps( device );
    this.atlas = new Atlas( device );

    // Trigger shader compilation before anything (will be cached)
    VelloShader.getShaders( device );

    this.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();

    // TODO: handle context losses, reconstruct with the device
    device.lost.then( info => {
      console.error( `WebGPU device was lost: ${info.message}` );

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
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    } );
    return context;
  }

  public static async create(): Promise<DeviceContext> {
    const adapter = await navigator.gpu?.requestAdapter( {
      powerPreference: 'high-performance'
    } );
    const device = await adapter?.requestDevice( {
        requiredFeatures: [ 'bgra8unorm-storage' ]
    } );
    if ( !device ) {
      throw new Error( 'need a browser that supports WebGPU' );
    }

    return new DeviceContext( device );
  }

  public dispose(): void {
    this.ramps.dispose();
    this.atlas.dispose();

    // TODO: destroy the device too?
  }
}

scenery.register( 'DeviceContext', DeviceContext );
