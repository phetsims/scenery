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
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';

export default class DeviceContext {

  public ramps: Ramps;
  public atlas: Atlas;
  public preferredCanvasFormat: GPUTextureFormat; // TODO: support other formats?

  public static currentDevice: GPUDevice | null = null;
  public static currentDeviceContext: DeviceContext | null = null;
  private static couldNotGetDevice = false;
  private static completedDeviceAttempt = false;

  public lostEmitter = new TinyEmitter();

  public constructor( public device: GPUDevice ) {
    this.ramps = new Ramps( device );
    this.atlas = new Atlas( device );

    // Trigger shader compilation before anything (will be cached)
    VelloShader.getShaders( device );

    this.preferredCanvasFormat = navigator.gpu.getPreferredCanvasFormat();

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
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,

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

    const adapter = await navigator.gpu?.requestAdapter( {
      powerPreference: 'high-performance'
    } );

    const device = await adapter?.requestDevice( {
      requiredFeatures: [ 'bgra8unorm-storage' ]
    } );

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

( async () => {
  const velloLock = asyncLoader.createLock( { name: 'vello' } );

  // This will cache it
  await DeviceContext.isVelloSupported();

  velloLock();
} )().catch( err => { throw err; } );

scenery.register( 'DeviceContext', DeviceContext );