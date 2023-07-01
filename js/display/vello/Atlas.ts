// Copyright 2023, University of Colorado Boulder

/**
 * Handles packing images into a texture atlas, and handles the texture/view.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { AtlasAllocator, AtlasBin, BufferImage, EncodableImage, scenery, VelloImagePatch } from '../../imports.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';

const ATLAS_INITIAL_SIZE = 256;
const ATLAS_MAX_SIZE = 8192;

// Amount of space along the edge of each image that is filled with the closest adjacent pixel value. This helps
// get rid of alpha fading, see https://github.com/phetsims/scenery/issues/637.
const GUTTER_SIZE = 1;

// Amount of blank space along the bottom and right of each image that is left transparent, to avoid graphical
// artifacts due to texture filtering blending the adjacent image in.
// See https://github.com/phetsims/scenery/issues/637.
const PADDING = 1;

let globalID = 0;

export default class Atlas {

  private id = globalID++;

  public width: number;
  public height: number;
  public texture: GPUTexture | null = null;
  public textureView: GPUTextureView | null = null;
  private allocator: AtlasAllocator;

  private dirtyAtlasSubImages: AtlasSubImage[] = [];
  private used: AtlasSubImage[] = [];
  private unused: AtlasSubImage[] = [];

  public constructor( public device: GPUDevice ) {
    // TODO: Do we have "repeat" on images also? Think repeating patterns!

    // TODO: atlas size (1) when no images?
    this.width = ATLAS_INITIAL_SIZE;
    this.height = ATLAS_INITIAL_SIZE;

    this.allocator = new AtlasAllocator( this.width, this.height );
    sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} created with ${this.width}x${this.height}` );

    this.replaceTexture();
  }

  public updatePatches( patches: VelloImagePatch[] ): void {

    // TODO: actually could we accomplish this with a generation?
    this.unused.push( ...this.used );
    this.used.length = 0;

    sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} image patches: ${patches.length}` );

    for ( let i = 0; i < patches.length; i++ ) {
      const patch = patches[ i ];
      patch.atlasSubImage = this.getAtlasSubImage( patch.image );
    }

    // It's possible that we "remapped" indices, so we need to update the atlasSubImages
    for ( let i = 0; i < patches.length; i++ ) {
      const patch = patches[ i ];
      if ( patch.atlasSubImage ) {
        patch.atlasSubImage.update();
      }
    }
  }

  // TODO: Add some "unique" identifier on BufferImage so we can check if it's actually representing the same image
  // TODO: would it kill performance to hash the image data? If we're changing the image every frame, that would
  // TODO: be very excessive.
  private getAtlasSubImage( image: EncodableImage ): AtlasSubImage | null {

    // Try a "used" one first (e.g. multiple in the same scene)
    for ( let i = 0; i < this.used.length; i++ ) {
      if ( this.used[ i ].image.equals( image ) ) {
        return this.used[ i ];
      }
    }

    // Try an "unused" one next (e.g. not yet in the scene)
    for ( let i = 0; i < this.unused.length; i++ ) {
      if ( this.unused[ i ].image.equals( image ) ) {
        const atlasSubImage = this.unused[ i ];
        this.unused.splice( this.unused.indexOf( atlasSubImage ), 1 );

        this.used.push( atlasSubImage );
        return atlasSubImage;
      }
    }

    // Drat, we'll need to allocate it
    const allocateWidth = image.width + GUTTER_SIZE * 2 + PADDING;
    const allocateHeight = image.height + GUTTER_SIZE * 2 + PADDING;
    assert && assert( isFinite( allocateWidth ) && isFinite( allocateHeight ) );
    sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} allocate ${allocateWidth}x${allocateHeight}` );

    // See if we have free space
    let bin = this.allocator.allocate( allocateWidth, allocateHeight );

    // Try after ditching unused images
    while ( !bin && this.unused.length ) {
      sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} deallocate` );
      this.allocator.deallocate( this.unused.pop()!.bin );

      bin = this.allocator.allocate( allocateWidth, allocateHeight );
    }

    // Try after resizing up
    let resized = false;
    while ( !bin && this.width < ATLAS_MAX_SIZE && this.height < ATLAS_MAX_SIZE ) {
      this.width *= 2;
      this.height *= 2;

      resized = true;
      sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} resizeAndRearrange ${this.width} ${this.height}` );
      this.allocator.resizeAndRearrange( this.width, this.height );
      bin = this.allocator.allocate( allocateWidth, allocateHeight );
      assert && this.verifyNoBinOverlap();
    }

    // Try after rearranging in the maximum size
    if ( !bin ) {
      resized = true;
      sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} REARRANGE!!!` );
      this.allocator.rearrange();
      bin = this.allocator.allocate( allocateWidth, allocateHeight );
      assert && this.verifyNoBinOverlap();
    }

    if ( resized ) {
      sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} updating size` );
      this.replaceTexture();
      this.dirtyAtlasSubImages.length = 0;
      this.dirtyAtlasSubImages.push( ...this.used );
    }

    if ( bin ) {
      const x = bin.x + GUTTER_SIZE;
      const y = bin.y + GUTTER_SIZE;

      const atlasSubImage = new AtlasSubImage( image, x, y, bin );
      this.dirtyAtlasSubImages.push( atlasSubImage );
      this.used.push( atlasSubImage );
      assert && this.verifyNoBinOverlap();
      return atlasSubImage;
    }
    else {
      sceneryLog && sceneryLog.Atlas && sceneryLog.Atlas( `#${this.id} FAILURE!!!!!!` );
      return null;
    }
  }

  private replaceTexture(): void {
    this.texture && this.texture.destroy();

    // TODO: Check this on Windows!
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    this.texture = this.device.createTexture( {
      label: 'atlas texture',
      size: {
        width: this.width || 1,
        height: this.height || 1,
        depthOrArrayLayers: 1
      },
      format: canvasFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    } );
    this.textureView = this.texture.createView( {
      label: 'atlas texture view',
      format: canvasFormat,
      dimension: '2d'
    } );
  }

  public updateTexture(): void {
    while ( this.dirtyAtlasSubImages.length ) {
      // TODO: copy in gutter! (so it's not fuzzy)

      const atlasSubImage = this.dirtyAtlasSubImages.pop()!;
      const image = atlasSubImage.image;

      // TODO: note premultiplied. Why we don't want to use BufferImages from canvas data
      if ( image instanceof BufferImage ) {
        // TODO: we have the ability to do this in a single call, would that be better ever for performance? Maybe a single
        // TODO: call if we have to update a bunch of sections at once?
        this.device.queue.writeTexture( {
          texture: this.texture!,
          origin: {
            x: atlasSubImage.x,
            y: atlasSubImage.y,
            z: 0
          }
        }, image.buffer, {
          offset: 0,
          bytesPerRow: image.width * 4
        }, {
          width: image.width,
          height: image.height,
          depthOrArrayLayers: 1
        } );
      }
      else {
        this.device.queue.copyExternalImageToTexture( {
          source: image.source
        }, {
          texture: this.texture!,
          origin: {
            x: atlasSubImage.x,
            y: atlasSubImage.y,
            z: 0
          },
          // Our ImageBitmap already should be premultiplied, so we don't want to premultiply it again
          premultipliedAlpha: false
        }, {
          width: image.width,
          height: image.height,
          depthOrArrayLayers: 1
        } );
      }
    }
  }

  // A few different ways, detects bin overlaps or other atlas issues
  private verifyNoBinOverlap(): void {
    if ( assert ) {
      const bins = [ ...this.used, ...this.unused ].map( atlasSubImage => atlasSubImage.bin );

      for ( let i = 0; i < bins.length; i++ ) {
        const bin = bins[ i ];
        for ( let j = i + 1; j < bins.length; j++ ) {
          const otherBin = bins[ j ];

          assert( !bin.bounds.intersection( otherBin.bounds ).hasNonzeroArea(), 'bins should not overlap' );
        }
      }

      const atlasSubImages = [ ...this.used, ...this.unused ];
      for ( let i = 0; i < atlasSubImages.length; i++ ) {
        const atlasSubImage = atlasSubImages[ i ];
        const bounds = new Bounds2( atlasSubImage.x, atlasSubImage.y, atlasSubImage.image.width, atlasSubImage.image.height );
        for ( let j = i + 1; j < atlasSubImages.length; j++ ) {
          const otherAtlasSubImage = atlasSubImages[ j ];
          const otherBounds = new Bounds2( otherAtlasSubImage.x, otherAtlasSubImage.y, otherAtlasSubImage.image.width, otherAtlasSubImage.image.height );

          assert( !bounds.intersection( otherBounds ).hasNonzeroArea(), 'images should not overlap' );
        }
      }
    }
  }

  // When dumping out atlas images, this creates a closure with the state when initially called (we'll need to apply it
  // later, when our atlas may have already changed).
  public getDebugPainter(): ( ( context: CanvasRenderingContext2D ) => void ) {
    const usedBounds = this.used.map( atlasSubImage => atlasSubImage.bin.bounds );
    const unusedBounds = this.unused.map( atlasSubImage => atlasSubImage.bin.bounds );

    return context => {
      context.strokeStyle = 'blue';
      unusedBounds.forEach( bounds => {
        context.strokeRect( bounds.x, bounds.y, bounds.width, bounds.height );
      } );
      context.strokeStyle = 'red';
      usedBounds.forEach( bounds => {
        context.strokeRect( bounds.x, bounds.y, bounds.width, bounds.height );
      } );
    };
  }

  public dispose(): void {
    this.allocator.dispose();

    this.texture && this.texture.destroy();
  }
}

scenery.register( 'Atlas', Atlas );

// TODO: pool these?
export class AtlasSubImage {
  public constructor(
    public readonly image: EncodableImage,
    public x: number,
    public y: number,
    public bin: AtlasBin
  ) {}

  public update(): void {
    this.x = this.bin.x + GUTTER_SIZE;
    this.y = this.bin.y + GUTTER_SIZE;
  }
}
