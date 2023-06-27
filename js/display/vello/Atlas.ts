// Copyright 2023, University of Colorado Boulder

/**
 * Handles packing images into a texture atlas, and handles the texture/view.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BinPacker, { Bin } from '../../../../dot/js/BinPacker.js';
import { BufferImage, EncodableImage, scenery, VelloImagePatch } from '../../imports.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';

const ATLAS_INITIAL_SIZE = 1024;
const ATLAS_MAX_SIZE = 8192;

// Amount of space along the edge of each image that is filled with the closest adjacent pixel value. This helps
// get rid of alpha fading, see https://github.com/phetsims/scenery/issues/637.
const GUTTER_SIZE = 1;

// Amount of blank space along the bottom and right of each image that is left transparent, to avoid graphical
// artifacts due to texture filtering blending the adjacent image in.
// See https://github.com/phetsims/scenery/issues/637.
const PADDING = 1;

export default class Atlas {

  public width: number;
  public height: number;
  public texture: GPUTexture | null = null;
  public textureView: GPUTextureView | null = null;
  private binPacker!: BinPacker;

  private dirtyAtlasSubImages: AtlasSubImage[] = [];
  private used: AtlasSubImage[] = [];
  private unused: AtlasSubImage[] = [];

  // TODO: better image atlas, something nice like https://github.com/nical/guillotiere?
  public constructor( public device: GPUDevice ) {
    // TODO: Do we have "repeat" on images also? Think repeating patterns!

    // TODO: atlas size (1) when no images?
    this.width = ATLAS_INITIAL_SIZE;
    this.height = ATLAS_INITIAL_SIZE;

    this.replaceTexture();
  }

  public updatePatches( patches: VelloImagePatch[] ): void {

    // TODO: actually could we accomplish this with a generation?
    this.unused.push( ...this.used );
    this.used.length = 0;

    // TODO: sort heuristic for bin packing, so we can find the largest ones first?

    for ( let i = 0; i < patches.length; i++ ) {
      const patch = patches[ i ];
      patch.atlasSubImage = this.getAtlasSubImage( patch.image );
    }
  }

  // TODO: Add some "unique" identifier on BufferImage so we can check if it's actually representing the same image
  // TODO: would it kill performance to hash the image data? If we're changing the image every frame, that would
  // TODO: be very excessive.
  private getAtlasSubImage( image: EncodableImage, existingAtlasSubImage?: AtlasSubImage ): AtlasSubImage | null {
    assert && assert( !existingAtlasSubImage || existingAtlasSubImage.image === image );

    for ( let i = 0; i < this.used.length; i++ ) {
      if ( this.used[ i ].image === image ) {
        assert && assert( !existingAtlasSubImage );
        return this.used[ i ];
      }
    }
    for ( let i = 0; i < this.unused.length; i++ ) {
      if ( this.unused[ i ].image === image ) {
        let atlasSubImage = this.unused[ i ];
        this.unused.splice( this.unused.indexOf( atlasSubImage ), 1 );

        // If we're switching it to used, we can swap instances (since outside things shouldn't have a reference)
        if ( existingAtlasSubImage ) {
          existingAtlasSubImage.bin = atlasSubImage.bin;
          existingAtlasSubImage.x = atlasSubImage.x;
          existingAtlasSubImage.y = atlasSubImage.y;
          atlasSubImage = existingAtlasSubImage;
        }

        this.used.push( atlasSubImage );
        return atlasSubImage;
      }
    }
    let bin: Bin | null = null;

    while ( !( bin = this.binPacker.allocate( image.width + GUTTER_SIZE * 2 + PADDING, image.height + GUTTER_SIZE * 2 + PADDING ) ) && this.unused.length ) {
      this.binPacker.deallocate( this.unused.pop()!.bin );
    }

    if ( bin ) {
      let atlasSubImage;
      const x = bin.bounds.minX + GUTTER_SIZE;
      const y = bin.bounds.minY + GUTTER_SIZE;
      if ( existingAtlasSubImage ) {
        existingAtlasSubImage.bin = bin;
        existingAtlasSubImage.x = x;
        existingAtlasSubImage.y = y;
        atlasSubImage = existingAtlasSubImage;
      }
      else {
        atlasSubImage = new AtlasSubImage( image, x, y, bin );
      }
      this.dirtyAtlasSubImages.push( atlasSubImage );
      this.used.push( atlasSubImage );
      return atlasSubImage;
    }
    else {
      // TODO: could try to repack it too
      if ( this.width === ATLAS_MAX_SIZE || this.height === ATLAS_MAX_SIZE ) {
        return null;
      }
      this.width *= 2;
      this.height *= 2;

      // Copy out the used images, so we can repack them
      const used = this.used.slice();
      this.used = [];
      this.unused.length = 0;
      this.dirtyAtlasSubImages.length = 0;

      this.replaceTexture();

      for ( let i = 0; i < used.length; i++ ) {
        const newSubImage = this.getAtlasSubImage( used[ i ].image, used[ i ] );

        assert && assert( newSubImage, 'We had these packed in the original, they should exist now' );
      }

      const atlasSubImage = this.getAtlasSubImage( image );
      assert && assert( atlasSubImage, 'Could not fit the new image in the texture atlas' );

      return atlasSubImage;
    }
  }

  private replaceTexture(): void {
    this.texture && this.texture.destroy();

    this.texture = this.device.createTexture( {
      label: 'atlas texture',
      size: {
        width: this.width || 1,
        height: this.height || 1,
        depthOrArrayLayers: 1
      },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    } );
    this.textureView = this.texture.createView( {
      label: 'atlas texture view',
      format: 'rgba8unorm',
      dimension: '2d'
    } );

    this.binPacker = new BinPacker( new Bounds2( 0, 0, this.width, this.height ) );

    // We'll need to redraw everything!
    this.dirtyAtlasSubImages = this.used.slice();
    this.unused = [];
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
        // TODO: Since ImageBitmaps are async, can we create an option somewhere to await them so we don't use stale data?
        this.device.queue.copyExternalImageToTexture( {
          source: image.source
        }, {
          texture: this.texture!,
          origin: {
            x: atlasSubImage.x,
            y: atlasSubImage.y,
            z: 0
          },
          premultipliedAlpha: true
        }, {
          width: image.width,
          height: image.height,
          depthOrArrayLayers: 1
        } );
      }

    }
  }

  public dispose(): void {
    this.texture && this.texture.destroy();
  }
}

scenery.register( 'Atlas', Atlas );

// TODO: pool these?
export class AtlasSubImage {
  public constructor(
    public image: EncodableImage,
    public x: number,
    public y: number,
    public bin: Bin
  ) {}
}
