// Copyright 2023, University of Colorado Boulder

/**
 * Handles Vello rendering, and associated resources (atlas, ramps) related to a GPUDevice
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Atlas, BufferPool, Ramps, RenderInfo, scenery, VelloShader } from '../../imports.js';
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';

export type PreferredCanvasFormat = 'bgra8unorm' | 'rgba8unorm';

const DEBUG_ATLAS = false;

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

  public render( renderInfo: RenderInfo, outTexture: GPUTexture ): void {
    const device = this.device;

    const onCompleteActions: ( () => void )[] = [];

    const shaders = VelloShader.getShaders( device );

    const renderConfig = renderInfo.renderConfig!;
    assert && assert( renderConfig );

    const sceneBytes = renderInfo.sceneBytes;
    const dispatchSizes = renderConfig.dispatchSizes;
    const bufferSizes = renderConfig.bufferSizes;
    const configBytes = renderConfig.configBytes;

    const bufferPool = new BufferPool( device );

    const sceneBuffer = bufferPool.getBuffer( sceneBytes.byteLength, 'scene buffer' );
    device.queue.writeBuffer( sceneBuffer, 0, sceneBytes.buffer );

    const configBuffer = device.createBuffer( {
      label: 'config buffer',
      size: configBytes.byteLength,

      // Different than the typical buffer from BufferPool, we'll create it here and manually destroy it
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    } );
    device.queue.writeBuffer( configBuffer, 0, configBytes.buffer );

    const infoBinDataBuffer = bufferPool.getBuffer( bufferSizes.bin_data.getSizeInBytes(), 'info_bin_data buffer' );
    const tileBuffer = bufferPool.getBuffer( bufferSizes.tiles.getSizeInBytes(), 'tile buffer' );
    const segmentsBuffer = bufferPool.getBuffer( bufferSizes.segments.getSizeInBytes(), 'segments buffer' );
    const ptclBuffer = bufferPool.getBuffer( bufferSizes.ptcl.getSizeInBytes(), 'ptcl buffer' );
    const reducedBuffer = bufferPool.getBuffer( bufferSizes.path_reduced.getSizeInBytes(), 'reduced buffer' );

    const encoder = device.createCommandEncoder( {
      label: 'the encoder'
    } );

    shaders.pathtag_reduce.dispatch( encoder, dispatchSizes.path_reduce, [
      configBuffer, sceneBuffer, reducedBuffer
    ] );

    let pathTagParentBuffer = reducedBuffer;

    let reduced2Buffer;
    let reducedScanBuffer;
    if ( dispatchSizes.useLargePathScan ) {
      reduced2Buffer = bufferPool.getBuffer( bufferSizes.path_reduced2.getSizeInBytes(), 'reduced2 buffer' );

      shaders.pathtag_reduce2.dispatch( encoder, dispatchSizes.path_reduce2, [
        reducedBuffer, reduced2Buffer
      ] );

      reducedScanBuffer = bufferPool.getBuffer( bufferSizes.path_reduced_scan.getSizeInBytes(), 'reducedScan buffer' );

      shaders.pathtag_scan1.dispatch( encoder, dispatchSizes.path_scan1, [
        reducedBuffer, reduced2Buffer, reducedScanBuffer
      ] );

      pathTagParentBuffer = reducedScanBuffer;
    }

    const tagmonoidBuffer = bufferPool.getBuffer( bufferSizes.path_monoids.getSizeInBytes(), 'tagmonoid buffer' );

    ( dispatchSizes.useLargePathScan ? shaders.pathtag_scan_large : shaders.pathtag_scan_small ).dispatch( encoder, dispatchSizes.path_scan, [
      configBuffer, sceneBuffer, pathTagParentBuffer, tagmonoidBuffer
    ] );

    bufferPool.freeBuffer( reducedBuffer );
    reduced2Buffer && bufferPool.freeBuffer( reduced2Buffer );
    reducedScanBuffer && bufferPool.freeBuffer( reducedScanBuffer );

    const pathBBoxBuffer = bufferPool.getBuffer( bufferSizes.path_bboxes.getSizeInBytes(), 'pathBBox buffer' );

    shaders.bbox_clear.dispatch( encoder, dispatchSizes.bbox_clear, [
      configBuffer, pathBBoxBuffer
    ] );

    const cubicBuffer = bufferPool.getBuffer( bufferSizes.cubics.getSizeInBytes(), 'cubic buffer' );

    shaders.pathseg.dispatch( encoder, dispatchSizes.path_seg, [
      configBuffer, sceneBuffer, tagmonoidBuffer, pathBBoxBuffer, cubicBuffer
    ] );

    const drawReducedBuffer = bufferPool.getBuffer( bufferSizes.draw_reduced.getSizeInBytes(), 'drawReduced buffer' );

    shaders.draw_reduce.dispatch( encoder, dispatchSizes.draw_reduce, [
      configBuffer, sceneBuffer, drawReducedBuffer
    ] );

    const drawMonoidBuffer = bufferPool.getBuffer( bufferSizes.draw_monoids.getSizeInBytes(), 'drawMonoid buffer' );

    const clipInpBuffer = bufferPool.getBuffer( bufferSizes.clip_inps.getSizeInBytes(), 'clipInp buffer' );

    shaders.draw_leaf.dispatch( encoder, dispatchSizes.draw_leaf, [
      configBuffer, sceneBuffer, drawReducedBuffer, pathBBoxBuffer, drawMonoidBuffer, infoBinDataBuffer, clipInpBuffer
    ] );

    bufferPool.freeBuffer( drawReducedBuffer );

    const clipElBuffer = bufferPool.getBuffer( bufferSizes.clip_els.getSizeInBytes(), 'clipEl buffer' );

    const clipBicBuffer = bufferPool.getBuffer( bufferSizes.clip_bics.getSizeInBytes(), 'clipBic buffer' );

    if ( dispatchSizes.clip_reduce.x > 0 ) {
      shaders.clip_reduce.dispatch( encoder, dispatchSizes.clip_reduce, [
        configBuffer, clipInpBuffer, pathBBoxBuffer, clipBicBuffer, clipElBuffer
      ] );
    }

    const clipBBoxBuffer = bufferPool.getBuffer( bufferSizes.clip_bboxes.getSizeInBytes(), 'clipBBox buffer' );

    if ( dispatchSizes.clip_leaf.x > 0 ) {
      shaders.clip_leaf.dispatch( encoder, dispatchSizes.clip_leaf, [
        configBuffer, clipInpBuffer, pathBBoxBuffer, clipBicBuffer, clipElBuffer, drawMonoidBuffer, clipBBoxBuffer
      ] );
    }

    bufferPool.freeBuffer( clipInpBuffer );
    bufferPool.freeBuffer( clipBicBuffer );
    bufferPool.freeBuffer( clipElBuffer );

    const drawBBoxBuffer = bufferPool.getBuffer( bufferSizes.draw_bboxes.getSizeInBytes(), 'drawBBox buffer' );

    const bumpBuffer = bufferPool.getBuffer( bufferSizes.bump_alloc.getSizeInBytes(), 'bump buffer' );

    const binHeaderBuffer = bufferPool.getBuffer( bufferSizes.bin_headers.getSizeInBytes(), 'binHeader buffer' );

    // TODO: wgpu might not have this implemented? Do I need a manual clear?
    // TODO: actually, we're not reusing the buffer, so it might be zero'ed out? Check spec
    // TODO: See if this clearBuffer is insufficient (implied by engine.rs docs)
    if ( encoder.clearBuffer ) {
      // NOTE: Firefox nightly didn't have clearBuffer, so we're feature-detecting it
      encoder.clearBuffer( bumpBuffer, 0 );
    }
    else {
      // TODO: can we avoid this, and just fresh-create the buffer every time?
      device.queue.writeBuffer( bumpBuffer, 0, new Uint8Array( bumpBuffer.size ) );
    }

    shaders.binning.dispatch( encoder, dispatchSizes.binning, [
      configBuffer, drawMonoidBuffer, pathBBoxBuffer, clipBBoxBuffer, drawBBoxBuffer, bumpBuffer, infoBinDataBuffer, binHeaderBuffer
    ] );

    bufferPool.freeBuffer( drawMonoidBuffer );
    bufferPool.freeBuffer( pathBBoxBuffer );
    bufferPool.freeBuffer( clipBBoxBuffer );

    // Note: this only needs to be rounded up because of the workaround to store the tile_offset
    // in storage rather than workgroup memory.
    const pathBuffer = bufferPool.getBuffer( bufferSizes.paths.getSizeInBytes(), 'path buffer' );

    shaders.tile_alloc.dispatch( encoder, dispatchSizes.tile_alloc, [
      configBuffer, sceneBuffer, drawBBoxBuffer, bumpBuffer, pathBuffer, tileBuffer
    ] );

    bufferPool.freeBuffer( drawBBoxBuffer );

    shaders.path_coarse_full.dispatch( encoder, dispatchSizes.path_coarse, [
      configBuffer, sceneBuffer, tagmonoidBuffer, cubicBuffer, pathBuffer, bumpBuffer, tileBuffer, segmentsBuffer
    ] );

    bufferPool.freeBuffer( tagmonoidBuffer );
    bufferPool.freeBuffer( cubicBuffer );

    shaders.backdrop_dyn.dispatch( encoder, dispatchSizes.backdrop, [
      configBuffer, pathBuffer, tileBuffer
    ] );

    shaders.coarse.dispatch( encoder, dispatchSizes.coarse, [
      configBuffer, sceneBuffer, drawMonoidBuffer, binHeaderBuffer, infoBinDataBuffer, pathBuffer, tileBuffer, bumpBuffer, ptclBuffer
    ] );

    // TODO: Check frees on all buffers. Note the config buffer (manually destroy that, or can we reuse it?)
    bufferPool.freeBuffer( sceneBuffer );
    bufferPool.freeBuffer( drawMonoidBuffer );
    bufferPool.freeBuffer( binHeaderBuffer );
    bufferPool.freeBuffer( pathBuffer );
    bufferPool.freeBuffer( bumpBuffer );


    this.ramps.updateTexture();
    this.atlas.updateTexture();

    // TODO: TS change so this is always defined
    const rampTextureView = this.ramps.textureView!;
    assert && assert( rampTextureView );

    const atlasTextureView = this.atlas.textureView!;
    assert && assert( atlasTextureView );

    if ( assert && DEBUG_ATLAS ) {
      const debugCanvas = document.createElement( 'canvas' );
      const debugContext = debugCanvas.getContext( 'webgpu' )!;
      debugCanvas.width = this.atlas.texture!.width;
      debugCanvas.height = this.atlas.texture!.height;
      debugContext.configure( {
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        alphaMode: 'premultiplied'
      } );

      shaders.blit.dispatch( encoder, debugContext.getCurrentTexture().createView(), atlasTextureView );

      const atlasPainter = this.atlas.getDebugPainter();

      onCompleteActions.push( () => {
        const canvas = document.createElement( 'canvas' );
        canvas.width = debugCanvas.width;
        canvas.height = debugCanvas.height;
        const context = canvas.getContext( '2d' )!;
        context.drawImage( debugCanvas, 0, 0 );

        atlasPainter( context );

        // TODO: This is getting cut off at a certain amount of pixels?
        console.log( canvas.toDataURL() );
      } );
    }

    // NOTE: This is relevant code for if we want to render to a different texture (how to create it)

    const canvasTextureFormat = outTexture.format;
    if ( canvasTextureFormat !== 'bgra8unorm' && canvasTextureFormat !== 'rgba8unorm' ) {
      throw new Error( 'unsupported format' );
    }

    const canOutputToCanvas = canvasTextureFormat === this.preferredStorageFormat;
    let fineOutputTextureView: GPUTextureView;
    let fineOutputTexture: GPUTexture | null = null;
    const outTextureView = outTexture.createView();

    if ( canOutputToCanvas ) {
      fineOutputTextureView = outTextureView;
    }
    else {
      fineOutputTexture = device.createTexture( {
        label: 'fineOutputTexture',
        size: {
          width: outTexture.width,
          height: outTexture.height,
          depthOrArrayLayers: 1
        },
        format: this.preferredStorageFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING // see TargetTexture
      } );
      fineOutputTextureView = fineOutputTexture.createView( {
        label: 'fineOutputTextureView',
        format: this.preferredStorageFormat,
        dimension: '2d'
      } );
    }

    // Have the fine-rasterization shader use the preferred format as output (for now)
    ( this.preferredStorageFormat === 'bgra8unorm' ? shaders.fine_bgra8unorm : shaders.fine_rgba8unorm ).dispatch( encoder, dispatchSizes.fine, [
      configBuffer, tileBuffer, segmentsBuffer, fineOutputTextureView, ptclBuffer, rampTextureView, infoBinDataBuffer, atlasTextureView
    ] );

    if ( !canOutputToCanvas ) {
      assert && assert( fineOutputTexture, 'If we cannot output to the Canvas directly, we will have created a texture' );

      shaders.blit.dispatch( encoder, outTextureView, fineOutputTextureView );
    }

    bufferPool.freeBuffer( tileBuffer );
    bufferPool.freeBuffer( segmentsBuffer );
    bufferPool.freeBuffer( ptclBuffer );
    bufferPool.freeBuffer( infoBinDataBuffer );

    const commandBuffer = encoder.finish();
    device.queue.submit( [ commandBuffer ] );

    // Conditionally listen to when the submitted work is done
    if ( onCompleteActions.length ) {
      device.queue.onSubmittedWorkDone().then( () => {
        onCompleteActions.forEach( action => action() );
      } ).catch( err => {
        throw err;
      } );
    }

    // for now TODO: can we reuse? Likely get some from reusing these
    configBuffer.destroy();
    fineOutputTexture && fineOutputTexture.destroy();

    bufferPool.dispose();
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
