// Copyright 2023, University of Colorado Boulder

/**
 * WebGPU code for rendering using the Vello shaders
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BufferPool, DeviceContext, RenderInfo, scenery, VelloShader } from '../../imports.js';

const DEBUG_ATLAS = false;

// TODO: name change!
const render = ( renderInfo: RenderInfo, deviceContext: DeviceContext, outTexture: GPUTexture ): void => {
  const device = deviceContext.device;

  const onCompleteActions: ( () => void )[] = [];

  const shaders = VelloShader.getShaders( device );

  const renderConfig = renderInfo.renderConfig!;
  assert && assert( renderConfig );

  const sceneBytes = renderInfo.packed;

  const workgroupCounts = renderConfig.workgroupCounts;
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

  shaders.pathtag_reduce.dispatch( encoder, workgroupCounts.path_reduce, [
    configBuffer, sceneBuffer, reducedBuffer
  ] );

  let pathTagParentBuffer = reducedBuffer;

  let reduced2Buffer;
  let reducedScanBuffer;
  if ( workgroupCounts.useLargePathScan ) {
    reduced2Buffer = bufferPool.getBuffer( bufferSizes.path_reduced2.getSizeInBytes(), 'reduced2 buffer' );

    shaders.pathtag_reduce2.dispatch( encoder, workgroupCounts.path_reduce2, [
      reducedBuffer, reduced2Buffer
    ] );

    reducedScanBuffer = bufferPool.getBuffer( bufferSizes.path_reduced_scan.getSizeInBytes(), 'reducedScan buffer' );

    shaders.pathtag_scan1.dispatch( encoder, workgroupCounts.path_scan1, [
      reducedBuffer, reduced2Buffer, reducedScanBuffer
    ] );

    pathTagParentBuffer = reducedScanBuffer;
  }

  const tagmonoidBuffer = bufferPool.getBuffer( bufferSizes.path_monoids.getSizeInBytes(), 'tagmonoid buffer' );

  ( workgroupCounts.useLargePathScan ? shaders.pathtag_scan_large : shaders.pathtag_scan_small ).dispatch( encoder, workgroupCounts.path_scan, [
    configBuffer, sceneBuffer, pathTagParentBuffer, tagmonoidBuffer
  ] );

  bufferPool.freeBuffer( reducedBuffer );
  reduced2Buffer && bufferPool.freeBuffer( reduced2Buffer );
  reducedScanBuffer && bufferPool.freeBuffer( reducedScanBuffer );

  const pathBBoxBuffer = bufferPool.getBuffer( bufferSizes.path_bboxes.getSizeInBytes(), 'pathBBox buffer' );

  shaders.bbox_clear.dispatch( encoder, workgroupCounts.bbox_clear, [
    configBuffer, pathBBoxBuffer
  ] );

  const cubicBuffer = bufferPool.getBuffer( bufferSizes.cubics.getSizeInBytes(), 'cubic buffer' );

  shaders.pathseg.dispatch( encoder, workgroupCounts.path_seg, [
    configBuffer, sceneBuffer, tagmonoidBuffer, pathBBoxBuffer, cubicBuffer
  ] );

  const drawReducedBuffer = bufferPool.getBuffer( bufferSizes.draw_reduced.getSizeInBytes(), 'drawReduced buffer' );

  shaders.draw_reduce.dispatch( encoder, workgroupCounts.draw_reduce, [
    configBuffer, sceneBuffer, drawReducedBuffer
  ] );

  const drawMonoidBuffer = bufferPool.getBuffer( bufferSizes.draw_monoids.getSizeInBytes(), 'drawMonoid buffer' );

  const clipInpBuffer = bufferPool.getBuffer( bufferSizes.clip_inps.getSizeInBytes(), 'clipInp buffer' );

  shaders.draw_leaf.dispatch( encoder, workgroupCounts.draw_leaf, [
    configBuffer, sceneBuffer, drawReducedBuffer, pathBBoxBuffer, drawMonoidBuffer, infoBinDataBuffer, clipInpBuffer
  ] );

  bufferPool.freeBuffer( drawReducedBuffer );

  const clipElBuffer = bufferPool.getBuffer( bufferSizes.clip_els.getSizeInBytes(), 'clipEl buffer' );

  const clipBicBuffer = bufferPool.getBuffer( bufferSizes.clip_bics.getSizeInBytes(), 'clipBic buffer' );

  if ( workgroupCounts.clip_reduce.x > 0 ) {
    shaders.clip_reduce.dispatch( encoder, workgroupCounts.clip_reduce, [
      configBuffer, clipInpBuffer, pathBBoxBuffer, clipBicBuffer, clipElBuffer
    ] );
  }

  const clipBBoxBuffer = bufferPool.getBuffer( bufferSizes.clip_bboxes.getSizeInBytes(), 'clipBBox buffer' );

  if ( workgroupCounts.clip_leaf.x > 0 ) {
    shaders.clip_leaf.dispatch( encoder, workgroupCounts.clip_leaf, [
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
    // NOTE: FIrefox nightly didn't have clearBuffer, so we're feature-detecting it
    encoder.clearBuffer( bumpBuffer, 0 );
  }
  else {
    // TODO: can we avoid this, and just fresh-create the buffer every time?
    device.queue.writeBuffer( bumpBuffer, 0, new Uint8Array( bumpBuffer.size ) );
  }

  shaders.binning.dispatch( encoder, workgroupCounts.binning, [
    configBuffer, drawMonoidBuffer, pathBBoxBuffer, clipBBoxBuffer, drawBBoxBuffer, bumpBuffer, infoBinDataBuffer, binHeaderBuffer
  ] );

  bufferPool.freeBuffer( drawMonoidBuffer );
  bufferPool.freeBuffer( pathBBoxBuffer );
  bufferPool.freeBuffer( clipBBoxBuffer );

  // Note: this only needs to be rounded up because of the workaround to store the tile_offset
  // in storage rather than workgroup memory.
  const pathBuffer = bufferPool.getBuffer( bufferSizes.paths.getSizeInBytes(), 'path buffer' );

  shaders.tile_alloc.dispatch( encoder, workgroupCounts.tile_alloc, [
    configBuffer, sceneBuffer, drawBBoxBuffer, bumpBuffer, pathBuffer, tileBuffer
  ] );

  bufferPool.freeBuffer( drawBBoxBuffer );

  shaders.path_coarse_full.dispatch( encoder, workgroupCounts.path_coarse, [
    configBuffer, sceneBuffer, tagmonoidBuffer, cubicBuffer, pathBuffer, bumpBuffer, tileBuffer, segmentsBuffer
  ] );

  bufferPool.freeBuffer( tagmonoidBuffer );
  bufferPool.freeBuffer( cubicBuffer );

  shaders.backdrop_dyn.dispatch( encoder, workgroupCounts.backdrop, [
    configBuffer, pathBuffer, tileBuffer
  ] );

  shaders.coarse.dispatch( encoder, workgroupCounts.coarse, [
    configBuffer, sceneBuffer, drawMonoidBuffer, binHeaderBuffer, infoBinDataBuffer, pathBuffer, tileBuffer, bumpBuffer, ptclBuffer
  ] );

  // TODO: Check frees on all buffers. Note the config buffer (manually destroy that, or can we reuse it?)
  bufferPool.freeBuffer( sceneBuffer );
  bufferPool.freeBuffer( drawMonoidBuffer );
  bufferPool.freeBuffer( binHeaderBuffer );
  bufferPool.freeBuffer( pathBuffer );
  bufferPool.freeBuffer( bumpBuffer );


  deviceContext.ramps.updateTexture();
  deviceContext.atlas.updateTexture();

  // TODO: TS change so this is always defined
  const rampTextureView = deviceContext.ramps.textureView!;
  assert && assert( rampTextureView );

  const atlasTextureView = deviceContext.atlas.textureView!;
  assert && assert( atlasTextureView );

  if ( assert && DEBUG_ATLAS ) {
    const debugCanvas = document.createElement( 'canvas' );
    const debugContext = debugCanvas.getContext( 'webgpu' )!;
    debugCanvas.width = deviceContext.atlas.texture!.width;
    debugCanvas.height = deviceContext.atlas.texture!.height;
    debugContext.configure( {
      device: deviceContext.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      alphaMode: 'premultiplied'
    } );

    shaders.blit.dispatch( encoder, debugContext.getCurrentTexture().createView(), atlasTextureView );

    const atlasPainter = deviceContext.atlas.getDebugPainter();

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

  const canOutputToCanvas = canvasTextureFormat === deviceContext.preferredStorageFormat;
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
      format: deviceContext.preferredStorageFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING // see TargetTexture
    } );
    fineOutputTextureView = fineOutputTexture.createView( {
      label: 'fineOutputTextureView',
      format: deviceContext.preferredStorageFormat,
      dimension: '2d'
    } );
  }

  // Have the fine-rasterization shader use the preferred format as output (for now)
  ( deviceContext.preferredStorageFormat === 'bgra8unorm' ? shaders.fine_bgra8unorm : shaders.fine_rgba8unorm ).dispatch( encoder, workgroupCounts.fine, [
    configBuffer, tileBuffer, segmentsBuffer, fineOutputTextureView, ptclBuffer, rampTextureView, infoBinDataBuffer, atlasTextureView
  ] );

  if ( !canOutputToCanvas ) {
    assert && assert( fineOutputTexture, 'If we cannot output to the Canvas directly, we will have created a texture' );

    shaders.blit.dispatch( encoder, outTextureView, fineOutputTextureView );
  }

  // TODO: are these early frees acceptable? Are we going to badly reuse things?
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
};

export default render;

scenery.register( 'render', render );