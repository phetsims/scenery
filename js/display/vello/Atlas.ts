// Copyright 2023, University of Colorado Boulder

/**
 * Handles packing images into a texture atlas, and handles the texture/view.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BufferImage } from './BufferImage.js';
import BinPacker, { Bin } from '../../../../dot/js/BinPacker.js';
import { scenery } from '../../imports.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import { VelloImagePatch } from './Encoding.js';

const ATLAS_INITIAL_SIZE = 2048; // TODO: revert to 1024, with sizing up (implement that!!)
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
  private binPacker: BinPacker;

  private dirtyAtlasSubImages: AtlasSubImage[] = [];
  private used: AtlasSubImage[] = [];
  private unused: AtlasSubImage[] = [];
  private generation = 0;

  // TODO: better image atlas, something nice like https://github.com/nical/guillotiere?
  public constructor( public device: GPUDevice ) {
    // TODO: Do we have "repeat" on images also? Think repeating patterns!

    // TODO: atlas size (1) when no images?
    this.width = ATLAS_INITIAL_SIZE;
    this.height = ATLAS_INITIAL_SIZE;
    this.binPacker = this.getBinPacker();

    this.replaceTexture();
  }

  public updatePatches( patches: VelloImagePatch[] ): void {
    const generation = this.generation++;

    // TODO: sort heuristic for bin packing, so we can find the largest ones first?

    patches.forEach( patch => {
      // TODO: reduce GC, no closures like this!!
      let atlasSubImage = _.find( this.used, atlasSubImage => atlasSubImage.image === patch.image );

      if ( !atlasSubImage ) {
        atlasSubImage = _.find( this.unused, atlasSubImage => atlasSubImage.image === patch.image );
        if ( atlasSubImage ) {
          this.used.push( atlasSubImage );
          this.unused.splice( this.unused.indexOf( atlasSubImage ), 1 );
        }
        else {
          let bin: Bin | null = null;

          while ( !( bin = this.binPacker.allocate( patch.image.width + GUTTER_SIZE * 2 + PADDING, patch.image.height + GUTTER_SIZE * 2 + PADDING ) ) && this.unused.length ) {
            this.binPacker.deallocate( this.unused.pop()!.bin );
          }

          if ( !bin ) {
            if ( this.width === ATLAS_MAX_SIZE || this.height === ATLAS_MAX_SIZE ) {
              throw new Error( 'maximum size atlas reached' );
            }
            this.width *= 2;
            this.height *= 2;
            this.replaceTexture();

            bin = this.binPacker.allocate( patch.image.width + GUTTER_SIZE * 2 + PADDING, patch.image.height + GUTTER_SIZE * 2 + PADDING );
          }

          atlasSubImage = new AtlasSubImage( patch.image, bin!.bounds.minX + GUTTER_SIZE, bin!.bounds.minY + GUTTER_SIZE, bin!, generation );
          this.dirtyAtlasSubImages.push( atlasSubImage );
          this.used.push( atlasSubImage );
        }
      }

      patch.atlasSubImage = atlasSubImage;

      if ( atlasSubImage.generation < generation ) {
        atlasSubImage.generation = generation;
      }
    } );
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
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    } );
    this.textureView = this.texture.createView( {
      label: 'atlas texture view',
      format: 'rgba8unorm',
      dimension: '2d'
    } );

    // TODO: REPACK everything!!!
    if ( this.binPacker && this.width > ATLAS_INITIAL_SIZE ) {
      throw new Error( 'resizing up unimplemented' );
    }
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
  }

  private getBinPacker(): BinPacker {
    return new BinPacker( new Bounds2( 0, 0, this.width, this.height ) );
  }

  public dispose(): void {
    this.texture && this.texture.destroy();
  }
}

scenery.register( 'Atlas', Atlas );

export class AtlasSubImage {
  public constructor(
    public image: BufferImage,
    public x: number,
    public y: number,
    public bin: Bin,
    public generation: number
  ) {}
}
