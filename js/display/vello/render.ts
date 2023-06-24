// Copyright 2023, University of Colorado Boulder

/**
 * WebGPU code for rendering using the Vello shaders
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BufferPool from './BufferPool.js';
import VelloShader from './VelloShader.js';
import DeviceContext from './DeviceContext.js';
import { RenderInfo } from './Encoding.js';
import { scenery } from '../../imports.js';

const render = ( renderInfo: RenderInfo, deviceContext: DeviceContext, outTexture: GPUTexture ): void => {
  const device = deviceContext.device;

  const shaders = VelloShader.getShaders( device );

  const preferredFormat = outTexture.format;
  if ( preferredFormat !== 'bgra8unorm' && preferredFormat !== 'rgba8unorm' ) {
    throw new Error( 'unsupported format' );
  }

  const renderConfig = renderInfo.renderConfig!;
  assert && assert( renderConfig );

  const sceneBytes = renderInfo.packed;

  const workgroupCounts = renderConfig.workgroup_counts;
  const bufferSizes = renderConfig.buffer_sizes;
  const configBytes = renderConfig.config_bytes;

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

  const infoBinDataBuffer = bufferPool.getBuffer( bufferSizes.bin_data.size_in_bytes(), 'info_bin_data buffer' );
  const tileBuffer = bufferPool.getBuffer( bufferSizes.tiles.size_in_bytes(), 'tile buffer' );
  const segmentsBuffer = bufferPool.getBuffer( bufferSizes.segments.size_in_bytes(), 'segments buffer' );
  const ptclBuffer = bufferPool.getBuffer( bufferSizes.ptcl.size_in_bytes(), 'ptcl buffer' );
  const reducedBuffer = bufferPool.getBuffer( bufferSizes.path_reduced.size_in_bytes(), 'reduced buffer' );

  const encoder = device.createCommandEncoder( {
    label: 'the encoder'
  } );

  shaders.pathtag_reduce.dispatch( encoder, workgroupCounts.path_reduce, [
    configBuffer, sceneBuffer, reducedBuffer
  ] );

  let pathTagParentBuffer = reducedBuffer;

  let reduced2Buffer;
  let reducedScanBuffer;
  if ( workgroupCounts.use_large_path_scan ) {
    reduced2Buffer = bufferPool.getBuffer( bufferSizes.path_reduced2.size_in_bytes(), 'reduced2 buffer' );

    shaders.pathtag_reduce2.dispatch( encoder, workgroupCounts.path_reduce2, [
      reducedBuffer, reduced2Buffer
    ] );

    reducedScanBuffer = bufferPool.getBuffer( bufferSizes.path_reduced_scan.size_in_bytes(), 'reducedScan buffer' );

    shaders.pathtag_scan1.dispatch( encoder, workgroupCounts.path_scan1, [
      reducedBuffer, reduced2Buffer, reducedScanBuffer
    ] );

    pathTagParentBuffer = reducedScanBuffer;
  }

  const tagmonoidBuffer = bufferPool.getBuffer( bufferSizes.path_monoids.size_in_bytes(), 'tagmonoid buffer' );

  ( workgroupCounts.use_large_path_scan ? shaders.pathtag_scan_large : shaders.pathtag_scan_small ).dispatch( encoder, workgroupCounts.path_scan, [
    configBuffer, sceneBuffer, pathTagParentBuffer, tagmonoidBuffer
  ] );

  bufferPool.freeBuffer( reducedBuffer );
  reduced2Buffer && bufferPool.freeBuffer( reduced2Buffer );
  reducedScanBuffer && bufferPool.freeBuffer( reducedScanBuffer );

  const pathBBoxBuffer = bufferPool.getBuffer( bufferSizes.path_bboxes.size_in_bytes(), 'pathBBox buffer' );

  shaders.bbox_clear.dispatch( encoder, workgroupCounts.bbox_clear, [
    configBuffer, pathBBoxBuffer
  ] );

  const cubicBuffer = bufferPool.getBuffer( bufferSizes.cubics.size_in_bytes(), 'cubic buffer' );

  shaders.pathseg.dispatch( encoder, workgroupCounts.path_seg, [
    configBuffer, sceneBuffer, tagmonoidBuffer, pathBBoxBuffer, cubicBuffer
  ] );

  const drawReducedBuffer = bufferPool.getBuffer( bufferSizes.draw_reduced.size_in_bytes(), 'drawReduced buffer' );

  shaders.draw_reduce.dispatch( encoder, workgroupCounts.draw_reduce, [
    configBuffer, sceneBuffer, drawReducedBuffer
  ] );

  const drawMonoidBuffer = bufferPool.getBuffer( bufferSizes.draw_monoids.size_in_bytes(), 'drawMonoid buffer' );

  const clipInpBuffer = bufferPool.getBuffer( bufferSizes.clip_inps.size_in_bytes(), 'clipInp buffer' );

  shaders.draw_leaf.dispatch( encoder, workgroupCounts.draw_leaf, [
    configBuffer, sceneBuffer, drawReducedBuffer, pathBBoxBuffer, drawMonoidBuffer, infoBinDataBuffer, clipInpBuffer
  ] );

  bufferPool.freeBuffer( drawReducedBuffer );

  const clipElBuffer = bufferPool.getBuffer( bufferSizes.clip_els.size_in_bytes(), 'clipEl buffer' );

  const clipBicBuffer = bufferPool.getBuffer( bufferSizes.clip_bics.size_in_bytes(), 'clipBic buffer' );

  if ( workgroupCounts.clip_reduce.x > 0 ) {
    shaders.clip_reduce.dispatch( encoder, workgroupCounts.clip_reduce, [
      configBuffer, clipInpBuffer, pathBBoxBuffer, clipBicBuffer, clipElBuffer
    ] );
  }

  const clipBBoxBuffer = bufferPool.getBuffer( bufferSizes.clip_bboxes.size_in_bytes(), 'clipBBox buffer' );

  if ( workgroupCounts.clip_leaf.x > 0 ) {
    shaders.clip_leaf.dispatch( encoder, workgroupCounts.clip_leaf, [
      configBuffer, clipInpBuffer, pathBBoxBuffer, clipBicBuffer, clipElBuffer, drawMonoidBuffer, clipBBoxBuffer
    ] );
  }

  bufferPool.freeBuffer( clipInpBuffer );
  bufferPool.freeBuffer( clipBicBuffer );
  bufferPool.freeBuffer( clipElBuffer );

  const drawBBoxBuffer = bufferPool.getBuffer( bufferSizes.draw_bboxes.size_in_bytes(), 'drawBBox buffer' );

  const bumpBuffer = bufferPool.getBuffer( bufferSizes.bump_alloc.size_in_bytes(), 'bump buffer' );

  const binHeaderBuffer = bufferPool.getBuffer( bufferSizes.bin_headers.size_in_bytes(), 'binHeader buffer' );


  // TODO: wgpu might not have this implemented? Do I need a manual clear?
  // TODO: actually, we're not reusing the buffer, so it might be zero'ed out? Check spec
  // TODO: See if this clearBuffer is insufficient (implied by engine.rs docs)
  encoder.clearBuffer( bumpBuffer, 0 );
  // device.queue.writeBuffer( bumpBuffer, 0, new Uint8Array( bumpBuffer.size ) );

  shaders.binning.dispatch( encoder, workgroupCounts.binning, [
    configBuffer, drawMonoidBuffer, pathBBoxBuffer, clipBBoxBuffer, drawBBoxBuffer, bumpBuffer, infoBinDataBuffer, binHeaderBuffer
  ] );

  bufferPool.freeBuffer( drawMonoidBuffer );
  bufferPool.freeBuffer( pathBBoxBuffer );
  bufferPool.freeBuffer( clipBBoxBuffer );

  // Note: this only needs to be rounded up because of the workaround to store the tile_offset
  // in storage rather than workgroup memory.
  const pathBuffer = bufferPool.getBuffer( bufferSizes.paths.size_in_bytes(), 'path buffer' );

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


  // NOTE: This is relevant code for if we want to render to a different texture (how to create it)
  // const outImage = device.createTexture( {
  //   label: 'outImage',
  //   size: {
  //     width: width,
  //     height: height,
  //     depthOrArrayLayers: 1
  //   },
  //   format: actualFormat,
  //   // TODO: wtf, usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
  //   usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
  // } );
  // const outImageView = outImage.createView( {
  //   label: 'outImageView',
  //   format: actualFormat,
  //   dimension: '2d'
  // } );

  deviceContext.ramps.updateTexture();
  deviceContext.atlas.updateTexture();

  const rampTextureView = deviceContext.ramps.textureView!;
  assert && assert( rampTextureView );

  const atlasTextureView = deviceContext.atlas.textureView!;
  assert && assert( atlasTextureView );

  // Have the fine-rasterization shader use the preferred format as output (for now)
  ( preferredFormat === 'bgra8unorm' ? shaders.fine_bgra8unorm : shaders.fine_rgba8unorm ).dispatch( encoder, workgroupCounts.fine, [
    configBuffer, tileBuffer, segmentsBuffer, outTexture.createView(), ptclBuffer, rampTextureView, infoBinDataBuffer, atlasTextureView
  ] );

  // NOTE: bgra8unorm vs rgba8unorm can't be copied, so this depends on the platform?
  // encoder.copyTextureToTexture( {
  //   texture: outImage
  // }, {
  //   texture: context.getCurrentTexture()
  // }, {
  //   width: width,
  //   height: height,
  //   depthOrArrayLayers: 1
  // } );

  // TODO: are these early frees acceptable? Are we going to badly reuse things?
  bufferPool.freeBuffer( tileBuffer );
  bufferPool.freeBuffer( segmentsBuffer );
  bufferPool.freeBuffer( ptclBuffer );
  bufferPool.freeBuffer( infoBinDataBuffer );

  const commandBuffer = encoder.finish();
  device.queue.submit( [ commandBuffer ] );
  // device.queue.onSubmittedWorkDone().then( () => {} );

  // for now TODO: can we reuse? Likely get some from reusing these
  configBuffer.destroy();

  bufferPool.dispose();
};

export default render;

scenery.register( 'render', render );
