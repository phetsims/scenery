// Copyright 2015-2022, University of Colorado Boulder

/**
 * A single Canvas/texture with multiple different images (sprites) drawn internally. During rendering, this texture
 * can be used in one draw call to render multiple different images by providing UV coordinates to each quad for each
 * image to be drawn.
 *
 * Note that the WebGL texture part is not required to be run - the Canvas-only part can be used functionally without
 * any WebGL dependencies.
 *
 * TODO: How to use custom mipmap levels?
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import BinPacker, { Bin } from '../../../dot/js/BinPacker.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Dimension2 from '../../../dot/js/Dimension2.js';
import { scenery } from '../imports.js';

// constants
// The max SpriteSheet size was selected to minimize memory overhead while still accommodating many large images
// See https://github.com/phetsims/scenery/issues/539
const MAX_DIMENSION = new Dimension2( 1024, 1024 );

// Amount of space along the edge of each image that is filled with the closest adjacent pixel value. This helps
// get rid of alpha fading, see https://github.com/phetsims/scenery/issues/637.
const GUTTER_SIZE = 1;

// Amount of blank space along the bottom and right of each image that is left transparent, to avoid graphical
// artifacts due to texture filtering blending the adjacent image in.
// See https://github.com/phetsims/scenery/issues/637.
const PADDING = 1;

export default class SpriteSheet {

  private useMipmaps: boolean;
  private gl: WebGLRenderingContext | null;
  public texture: WebGLTexture | null;

  // The top-level bounding box for texture content. All sprites will have coordinate bounding
  // boxes that are included in these bounds.
  private bounds: Bounds2;

  private width: number;
  private height: number;

  private canvas: HTMLCanvasElement;
  private context: CanvasRenderingContext2D;

  // Handles how our available area is partitioned into sprites.
  private binPacker: BinPacker;

  // Whether this spritesheet needs updates.
  private dirty: boolean;

  private usedSprites: Sprite[];

  // works as a LRU cache for removing items when we need to allocate new space
  private unusedSprites: Sprite[];

  /**
   * @param useMipmaps - Whether built-in WebGL mipmapping should be used. Higher quality, but may be slower
   *                     to add images (since mipmaps need to be updated).
   */
  public constructor( useMipmaps: boolean ) {
    this.useMipmaps = useMipmaps;

    // Will be passed in with initializeContext
    this.gl = null;

    // Will be set later, once we have a context
    this.texture = null;

    // TODO: potentially support larger texture sizes based on reported capabilities (could cause fewer draw calls?)
    this.bounds = new Bounds2( 0, 0, MAX_DIMENSION.width, MAX_DIMENSION.height );
    assert && assert( this.bounds.minX === 0 && this.bounds.minY === 0, 'Assumed constraint later on for transforms' );

    this.width = this.bounds.width;
    this.height = this.bounds.height;

    this.canvas = document.createElement( 'canvas' );
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.context = this.canvas.getContext( '2d' )!;

    this.binPacker = new BinPacker( this.bounds );
    this.dirty = true;

    this.usedSprites = [];
    this.unusedSprites = [];
  }

  /**
   * Initialize (or reinitialize) ourself with a new GL context. Should be called at least once before updateTexture()
   *
   * NOTE: Should be safe to call with a different context (will recreate a different texture) should this be needed
   *       for things like context loss.
   */
  public initializeContext( gl: WebGLRenderingContext ): void {
    this.gl = gl;

    this.createTexture();
  }

  /**
   * Allocates and creates a GL texture, configures it, and initializes it with our current Canvas.
   */
  private createTexture(): void {
    const gl = this.gl!;
    assert && assert( gl );

    this.texture = gl.createTexture();
    gl.bindTexture( gl.TEXTURE_2D, this.texture );
    gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
    gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
    gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, this.useMipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR );
    gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
    gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, false );
    // NOTE: We switched back to a fully premultiplied setup, and we were running into issues with the default
    // filtering/interpolation EXPECTING the texture ITSELF to be premultipled to work correctly (particularly with
    // textures that are larger or smaller on the screen).
    // See https://github.com/phetsims/energy-skate-park/issues/39, https://github.com/phetsims/scenery/issues/397
    // and https://stackoverflow.com/questions/39341564/webgl-how-to-correctly-blend-alpha-channel-png
    gl.pixelStorei( gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true ); // work with premultiplied numbers
    gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas );
    if ( this.useMipmaps ) {
      gl.hint( gl.GENERATE_MIPMAP_HINT, gl.NICEST );
      gl.generateMipmap( gl.TEXTURE_2D );
    }
    gl.bindTexture( gl.TEXTURE_2D, null );

    this.dirty = false;
  }

  /**
   * Updates a pre-existing texture with our current Canvas.
   */
  public updateTexture(): void {
    assert && assert( this.gl, 'SpriteSheet needs context to updateTexture()' );

    if ( this.dirty ) {
      this.dirty = false;

      const gl = this.gl!;
      assert && assert( gl );

      gl.bindTexture( gl.TEXTURE_2D, this.texture );
      gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.canvas );
      if ( this.useMipmaps ) {
        gl.hint( gl.GENERATE_MIPMAP_HINT, gl.NICEST );
        gl.generateMipmap( gl.TEXTURE_2D );
      }
      gl.bindTexture( gl.TEXTURE_2D, null );
    }
  }

  /**
   * Adds an image (if possible) to our sprite sheet. If successful, will return a {Sprite}, otherwise null.
   *
   * @param image
   * @param width - Passed in, since it may not be fully loaded yet?
   * @param height - Passed in, since it may not be fully loaded yet?
   */
  public addImage( image: HTMLCanvasElement | HTMLImageElement, width: number, height: number ): Sprite | null {
    let i;

    // check used cache
    for ( i = 0; i < this.usedSprites.length; i++ ) {
      const usedSprite = this.usedSprites[ i ];
      if ( usedSprite.image === image ) {
        usedSprite.count++;
        return usedSprite;
      }
    }

    // check unused cache
    for ( i = 0; i < this.unusedSprites.length; i++ ) {
      const unusedSprite = this.unusedSprites[ i ];
      if ( unusedSprite.image === image ) {
        unusedSprite.count++;
        assert && assert( unusedSprite.count === 1, 'Count should be exactly 1 after coming back from being unused' );
        this.unusedSprites.splice( i, 1 ); // remove it from the unused array
        this.usedSprites.push( unusedSprite ); // add it to the used array
        return unusedSprite;
      }
    }

    // Not in any caches, let's try to find space. If we can't find space at first, we start removing unused sprites
    // one-by-one.
    let bin;
    // Enters 'while' loop only if allocate() returns null and we have unused sprites (i.e. conditions where we will
    // want to deallocate the least recently used (LRU) unused sprite and then check for allocation again).
    while ( !( bin = this.binPacker.allocate( width + 2 * GUTTER_SIZE + PADDING, height + 2 * GUTTER_SIZE + PADDING ) ) && this.unusedSprites.length ) {
      const ejectedSprite = this.unusedSprites.shift()!; // LRU policy by taking first item

      // clear its space in the Canvas
      this.dirty = true;
      const ejectedBounds = ejectedSprite.bin.bounds;
      this.context.clearRect( ejectedBounds.x, ejectedBounds.y, ejectedBounds.width, ejectedBounds.height );

      // deallocate its area in the bin packer
      this.binPacker.deallocate( ejectedSprite.bin );
    }

    if ( bin ) {
      // WebGL will want UV coordinates in the [0,1] range
      // We need to chop off the gutters (on all sides), and the padding (on the bottom and right)
      const uvBounds = new Bounds2(
        ( bin.bounds.minX + GUTTER_SIZE ) / this.width,
        ( bin.bounds.minY + GUTTER_SIZE ) / this.height,
        ( bin.bounds.maxX - GUTTER_SIZE - PADDING ) / this.width,
        ( bin.bounds.maxY - GUTTER_SIZE - PADDING ) / this.height );
      const sprite = new Sprite( this, bin, uvBounds, image, 1 );

      this.copyImageWithGutter( image, width, height, bin.bounds.x, bin.bounds.y );

      this.dirty = true;
      this.usedSprites.push( sprite );
      return sprite;
    }
    // no space, even after clearing out our unused sprites
    else {
      return null;
    }
  }

  /**
   * Removes an image from our spritesheet. (Removes one from the amount it is used, and if it is 0, gets actually
   * removed).
   */
  public removeImage( image: HTMLCanvasElement | HTMLImageElement ): void {
    // find the used sprite (and its index)
    let usedSprite: Sprite;
    let i;
    for ( i = 0; i < this.usedSprites.length; i++ ) {
      if ( this.usedSprites[ i ].image === image ) {
        usedSprite = this.usedSprites[ i ];
        break;
      }
    }
    assert && assert( usedSprite!, 'Sprite not found for removeImage' );

    // if we have no more references to the image/sprite
    if ( --usedSprite!.count <= 0 ) {
      this.usedSprites.splice( i, 1 ); // remove it from the used list
      this.unusedSprites.push( usedSprite! ); // add it to the unused list
    }

    // NOTE: no modification to the Canvas/texture is made, since we can leave it drawn there and unreferenced.
    // If addImage( image ) is called for the same image, we can 'resurrect' it without any further Canvas/texture
    // changes being made.
  }

  /**
   * Whether the sprite for the specified image is handled by this spritesheet. It can be either used or unused, but
   * addImage() calls with the specified image should be extremely fast (no need to modify the Canvas or texture).
   */
  public containsImage( image: HTMLCanvasElement | HTMLImageElement ): boolean {
    let i;

    // check used cache
    for ( i = 0; i < this.usedSprites.length; i++ ) {
      if ( this.usedSprites[ i ].image === image ) {
        return true;
      }
    }

    // check unused cache
    for ( i = 0; i < this.unusedSprites.length; i++ ) {
      if ( this.unusedSprites[ i ].image === image ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Copes the image (width x height) centered into a bin (width+2 x height+2) at (binX,binY), where the padding
   * along the edges is filled with the next closest pixel in the actual image.
   */
  private copyImageWithGutter( image: HTMLCanvasElement | HTMLImageElement, width: number, height: number, binX: number, binY: number ): void {
    assert && assert( GUTTER_SIZE === 1 );

    // Corners, all 1x1
    this.copyImageRegion( image, 1, 1, 0, 0, binX, binY );
    this.copyImageRegion( image, 1, 1, width - 1, 0, binX + 1 + width, binY );
    this.copyImageRegion( image, 1, 1, width - 1, height - 1, binX + 1 + width, binY + 1 + height );
    this.copyImageRegion( image, 1, 1, 0, height - 1, binX, binY + 1 + height );

    // Edges
    this.copyImageRegion( image, width, 1, 0, 0, binX + 1, binY );
    this.copyImageRegion( image, width, 1, 0, height - 1, binX + 1, binY + 1 + height );
    this.copyImageRegion( image, 1, height, 0, 0, binX, binY + 1 );
    this.copyImageRegion( image, 1, height, width - 1, 0, binX + 1 + width, binY + 1 );

    this.context.drawImage( image, binX + 1, binY + 1 );
  }

  /**
   * Helper for drawing gutters.
   */
  private copyImageRegion( image: HTMLCanvasElement | HTMLImageElement, width: number, height: number, sourceX: number, sourceY: number, destinationX: number, destinationY: number ): void {
    this.context.drawImage( image, sourceX, sourceY, width, height, destinationX, destinationY, width, height );
  }

  public static Sprite: typeof Sprite;

  // the size of a sprite sheet
  public static readonly MAX_DIMENSION = MAX_DIMENSION;
}

scenery.register( 'SpriteSheet', SpriteSheet );

class Sprite {

  // The containing SpriteSheet
  public readonly spriteSheet: SpriteSheet;

  // Contains the actual image bounds in our Canvas (plus padding), and is used to deallocate (need to clear that area).
  // (dot-internal)
  public readonly bin: Bin;

  // Normalized bounds between [0,1] for the full texture (for GLSL texture lookups).
  public readonly uvBounds: Bounds2;

  // Image element used. (dot-internal)
  public readonly image: HTMLCanvasElement | HTMLImageElement;

  // Reference count for number of addChild() calls minus removeChild() calls. If the count is 0, it should be in the
  // 'unusedSprites' array, otherwise it should be in the 'usedSprites' array. (dot-internal)
  public count: number;

  /**
   * A reference to a specific part of the texture that can be used.
   */
  public constructor( spriteSheet: SpriteSheet, bin: Bin, uvBounds: Bounds2, image: HTMLCanvasElement | HTMLImageElement, initialCount: number ) {
    this.spriteSheet = spriteSheet;
    this.bin = bin;
    this.uvBounds = uvBounds;
    this.image = image;
    this.count = initialCount;
  }
}

SpriteSheet.Sprite = Sprite;