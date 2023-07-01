// Copyright 2023, University of Colorado Boulder

/**
 * Handles resources (atlas, ramps) related to a GPUDevice
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Atlas, Ramps, scenery, VelloShader } from '../../imports.js';
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';

export type PreferredCanvasFormat = 'bgra8unorm' | 'rgba8unorm';

export default class DeviceContext {

  public readonly ramps: Ramps;
  public readonly atlas: Atlas;
  public readonly preferredCanvasFormat: PreferredCanvasFormat;
  public readonly preferredStorageFormat: 'bgra8unorm' | 'rgba8unorm';

  public static currentDevice: GPUDevice | null = null;
  public static supportsBGRATextureStorage = false;
  private static couldNotGetDevice = false;
  private static completedDeviceAttempt = false;

  public lostEmitter = new TinyEmitter();

  private constructor( public device: GPUDevice ) {
    this.ramps = new Ramps( device );
    this.atlas = new Atlas( device );

    this.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat() as PreferredCanvasFormat;
    assert && assert( this.preferredCanvasFormat === 'bgra8unorm' || this.preferredCanvasFormat === 'rgba8unorm',
      'According to WebGPU documentation, this should only be bgra8unorm or rgba8unorm' );

    this.preferredStorageFormat = ( this.preferredCanvasFormat === 'bgra8unorm' && device.features.has( 'bgra8unorm-storage' ) )
                                  ? 'bgra8unorm'
                                  : 'rgba8unorm';

    // TODO: handle context losses, reconstruct with the device
    // TODO: get setup to manually trigger context losses
    // TODO: If the GPU is unavailable, we will return ALREADY LOST contexts. We should try an immediate request for a
    // TODO: device once, to see if we get a context back (transient loss), otherwise disable it for a while
    device.lost.then( info => {
      console.error( `WebGPU device was lost: ${info.message}` );

      this.lostEmitter.emit();

      DeviceContext.currentDevice = null;

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

  public static async isVelloSupported(): Promise<boolean> {
    // We want to make sure our shaders are validating AND our atlas/ramp initial texture code is working, so we
    // await a full device context here.
    return !!await DeviceContext.getDeviceContext();
  }

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

      try {
        await VelloShader.getShadersWithValidation( device );
      }
      catch( err ) {
        console.log( 'WebGPU validation error' );
        console.log( err );

        // Null things out!
        DeviceContext.couldNotGetDevice = true;
        DeviceContext.currentDevice = null;

        return null;
      }
    }
    else {
      DeviceContext.couldNotGetDevice = true;
    }

    return device || null;
  }

  // Just try to get a device context synchronously
  public static getSync(): DeviceContext | null {
    assert && assert( DeviceContext.completedDeviceAttempt, 'We should have awaited isVelloSupported() before calling this' );

    if ( DeviceContext.currentDevice ) {
      return new DeviceContext( DeviceContext.currentDevice );
    }
    else {
      return null;
    }
  }

  public static async getDeviceContext(): Promise<DeviceContext | null> {
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
