// Copyright 2023, University of Colorado Boulder

/**
 * Handles resources related to a GPUDevice
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
  public static currentDeviceContext: DeviceContext | null = null;
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
    return !!await DeviceContext.getDeviceContext();
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

      try {
        device.pushErrorScope( 'validation' );

        // Trigger shader compilation before anything (will be cached)
        VelloShader.getShaders( device );

        ( async () => {
          const validationError = await device.popErrorScope();

          if ( validationError ) {
            console.error( 'WebGPU validation error:', validationError );
            // TODO: this delayed action... might not be soon enough?
            // Null things out!
            DeviceContext.couldNotGetDevice = true;
            DeviceContext.currentDevice = null;
            DeviceContext.currentDeviceContext = null;
          }
        } )().catch( err => { throw err; } );
      }
      catch( err ) {
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

  public static async getDeviceContext(): Promise<DeviceContext | null> {
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
