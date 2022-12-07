// Copyright 2020-2022, University of Colorado Boulder

/**
 * Isolates Image handling with HTML/Canvas images, with mipmaps and general support.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Utils from '../../../dot/js/Utils.js';
import { Shape } from '../../../kite/js/imports.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, svgns, xlinkns } from '../imports.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import TEmitter from '../../../axon/js/TEmitter.js';

// Need to poly-fill on some browsers
const log2 = Math.log2 || function( x: number ) { return Math.log( x ) / Math.LN2; };

const DEFAULT_OPTIONS = {
  imageOpacity: 1,
  initialWidth: 0,
  initialHeight: 0,
  mipmap: false,
  mipmapBias: 0,
  mipmapInitialLevel: 4,
  mipmapMaxLevel: 5,
  hitTestPixels: false
} as const;

// Lazy scratch canvas/context (so we don't incur the startup cost of canvas/context creation)
let scratchCanvas: HTMLCanvasElement | null = null;
let scratchContext: CanvasRenderingContext2D | null = null;
const getScratchCanvas = (): HTMLCanvasElement => {
  if ( !scratchCanvas ) {
    scratchCanvas = document.createElement( 'canvas' );
  }
  return scratchCanvas;
};
const getScratchContext = () => {
  if ( !scratchContext ) {
    scratchContext = getScratchCanvas().getContext( '2d' )!;
  }
  return scratchContext;
};

type Constructor<T = object> = new ( ...args: IntentionalAny[] ) => T;

export type Mipmap = {
  width: number;
  height: number;
  url: string;
  canvas?: HTMLCanvasElement;
  img?: HTMLImageElement;
  updateCanvas?: () => void;
}[];

export type ImageableImage = string | HTMLImageElement | HTMLCanvasElement | Mipmap;

export type ImageableOptions = {
  image?: ImageableImage;
  imageOpacity?: number;
  initialWidth?: number;
  initialHeight?: number;
  mipmap?: boolean;
  mipmapBias?: number;
  mipmapInitialLevel?: number;
  mipmapMaxLevel?: number;
  hitTestPixels?: boolean;
};

const Imageable = <SuperType extends Constructor>( type: SuperType ) => { // eslint-disable-line @typescript-eslint/explicit-module-boundary-types
  return class ImageableMixin extends type {

    // (scenery-internal) Internal stateful value, see setImage()
    public _image: HTMLImageElement | HTMLCanvasElement | null;

    // Internal stateful value, see setInitialWidth() for documentation.
    private _initialWidth: number;

    // Internal stateful value, see setInitialHeight() for documentation.
    private _initialHeight: number;

    // (scenery-internal) Internal stateful value, see setImageOpacity() for documentation.
    public _imageOpacity: number;

    // (scenery-internal) Internal stateful value, see setMipmap() for documentation.
    public _mipmap: boolean;

    // (scenery-internal) Internal stateful value, see setMipmapBias() for documentation.
    public _mipmapBias: number;

    // (scenery-internal) Internal stateful value, see setMipmapInitialLevel() for documentation.
    public _mipmapInitialLevel: number;

    // (scenery-internal) Internal stateful value, see setMipmapMaxLevel() for documentation
    public _mipmapMaxLevel: number;

    // Internal stateful value, see setHitTestPixels() for documentation
    protected _hitTestPixels: boolean;

    // Array of Canvases for each level, constructed internally so that Canvas-based drawables (Canvas, WebGL) can quickly draw mipmaps.
    private _mipmapCanvases: HTMLCanvasElement[];

    // Array of URLs for each level, where each URL will display an image (and is typically a data URI or blob URI), so
    // that we can handle mipmaps in SVG where URLs are required.
    private _mipmapURLs: string[];

    // (scenery-internal) Mipmap data if it is passed into our image. Will be stored here for processing
    public _mipmapData: Mipmap | null;

    // Listener for invalidating our bounds whenever an image is invalidated.
    private _imageLoadListener: () => void;

    // Whether our _imageLoadListener has been attached as a listener to the current image.
    private _imageLoadListenerAttached: boolean;

    // Used for pixel hit testing.
    protected _hitTestImageData: ImageData | null;

    // Emits when mipmaps are (re)generated
    public mipmapEmitter: TEmitter;

    // For compatibility
    public isDisposed?: boolean;

    public constructor( ...args: IntentionalAny[] ) {

      super( ...args );

      this._image = null;
      this._initialWidth = DEFAULT_OPTIONS.initialWidth;
      this._initialHeight = DEFAULT_OPTIONS.initialHeight;
      this._imageOpacity = DEFAULT_OPTIONS.imageOpacity;
      this._mipmap = DEFAULT_OPTIONS.mipmap;
      this._mipmapBias = DEFAULT_OPTIONS.mipmapBias;
      this._mipmapInitialLevel = DEFAULT_OPTIONS.mipmapInitialLevel;
      this._mipmapMaxLevel = DEFAULT_OPTIONS.mipmapMaxLevel;
      this._hitTestPixels = DEFAULT_OPTIONS.hitTestPixels;
      this._mipmapCanvases = [];
      this._mipmapURLs = [];
      this._mipmapData = null;
      this._imageLoadListener = this._onImageLoad.bind( this );
      this._imageLoadListenerAttached = false;
      this._hitTestImageData = null;
      this.mipmapEmitter = new TinyEmitter();
    }


    /**
     * Sets the current image to be displayed by this Image node.
     *
     * We support a few different 'image' types that can be passed in:
     *
     * HTMLImageElement - A normal HTML <img>. If it hasn't been fully loaded yet, Scenery will take care of adding a
     *   listener that will update Scenery with its width/height (and load its data) when the image is fully loaded.
     *   NOTE that if you just created the <img>, it probably isn't loaded yet, particularly in Safari. If the Image
     *   node is constructed with an <img> that hasn't fully loaded, it will have a width and height of 0, which may
     *   cause issues if you are using bounds for layout. Please see initialWidth/initialHeight notes below.
     *
     * URL - Provide a {string}, and Scenery will assume it is a URL. This can be a normal URL, or a data URI, both will
     *   work. Please note that this has the same loading-order issues as using HTMLImageElement, but that it's almost
     *   always guaranteed to not have a width/height when you create the Image node. Note that data URI support for
     *   formats depends on the browser - only JPEG and PNG are supported broadly. Please see initialWidth/initialHeight
     *   notes below.
     *   Additionally, note that if a URL is provided, accessing image.getImage() or image.image will result not in the
     *   original URL (currently), but with the automatically created HTMLImageElement.
     *   TODO: return the original input
     *
     * HTMLCanvasElement - It's possible to pass an HTML5 Canvas directly into the Image node. It will immediately be
     *   aware of the width/height (bounds) of the Canvas, but NOTE that the Image node will not listen to Canvas size
     *   changes. It is assumed that after you pass in a Canvas to an Image node that it will not be modified further.
     *   Additionally, the Image node will only be rendered using Canvas or WebGL if a Canvas is used as input.
     *
     * Mipmap data structure - Image supports a mipmap data structure that provides rasterized mipmap levels. The 'top'
     *   level (level 0) is the entire full-size image, and every other level is twice as small in every direction
     *   (~1/4 the pixels), rounding dimensions up. This is useful for browsers that display the image badly if the
     *   image is too large. Instead, Scenery will dynamically pick the most appropriate size of the image to use,
     *   which improves the image appearance.
     *   The passed in 'image' should be an Array of mipmap objects of the format:
     *   {
     *     img: {HTMLImageElement}, // preferably preloaded, but it isn't required
     *     url: {string}, // URL (usually a data URL) for the image level
     *     width: {number}, // width of the mipmap level, in pixels
     *     height: {number} // height of the mipmap level, in pixels,
     *     canvas: {HTMLCanvasElement} // Canvas element containing the image data for the img.
     *     [updateCanvas]: {function} // If available, should be called before using the Canvas directly.
     *   }
     *   At least one level is required (level 0), and each mipmap level corresponds to the index in the array, e.g.:
     *   [
     *     level 0 (full size, e.g. 100x64)
     *     level 1 (half size, e.g. 50x32)
     *     level 2 (quarter size, e.g. 25x16)
     *     level 3 (eighth size, e.g. 13x8 - note the rounding up)
     *     ...
     *     level N (single pixel, e.g. 1x1 - this is the smallest level permitted, and there should only be one)
     *   ]
     *   Additionally, note that (currently) image.getImage() will return the HTMLImageElement from the first level,
     *   not the mipmap data.
     *   TODO: return the original input
     *
     *  Also note that if the underlying image (like Canvas data) has changed, it is recommended to call
     *  invalidateImage() instead of changing the image reference (calling setImage() multiple times)
     */
    public setImage( image: ImageableImage ): this {
      assert && assert( image, 'image should be available' );

      // Generally, if a different value for image is provided, it has changed
      let hasImageChanged = this._image !== image;

      // Except in some cases, where the provided image is a string. If our current image has the same .src as the
      // "new" image, it's basically the same (as we promote string images to HTMLImageElements).
      if ( hasImageChanged && typeof image === 'string' && this._image && this._image instanceof HTMLImageElement && image === this._image.src ) {
        hasImageChanged = false;
      }

      // Or if our current mipmap data is the same as the input, then we aren't changing it
      if ( hasImageChanged && image === this._mipmapData ) {
        hasImageChanged = false;
      }

      if ( hasImageChanged ) {
        // Reset the initial dimensions, since we have a new image that may have different dimensions.
        this._initialWidth = 0;
        this._initialHeight = 0;

        // Don't leak memory by referencing old images
        if ( this._image && this._imageLoadListenerAttached ) {
          this._detachImageLoadListener();
        }

        // clear old mipmap data references
        this._mipmapData = null;

        // Convert string => HTMLImageElement
        if ( typeof image === 'string' ) {
          // create an image with the assumed URL
          const src = image;
          image = document.createElement( 'img' );
          image.src = src;
        }
        // Handle the provided mipmap
        else if ( Array.isArray( image ) ) {
          // mipmap data!
          this._mipmapData = image;
          image = image[ 0 ].img!; // presumes we are already loaded

          // force initialization of mipmapping parameters, since invalidateMipmaps() is guaranteed to run below
          this._mipmapInitialLevel = this._mipmapMaxLevel = this._mipmapData.length;
          this._mipmap = true;
        }

        // We ruled out the string | Mipmap cases above
        this._image = image;

        // If our image is an HTML image that hasn't loaded yet, attach a load listener.
        if ( this._image instanceof HTMLImageElement && ( !this._image.width || !this._image.height ) ) {
          this._attachImageLoadListener();
        }

        // Try recomputing bounds (may give a 0x0 if we aren't yet loaded)
        this.invalidateImage();
      }
      return this;
    }

    public set image( value: ImageableImage ) { this.setImage( value ); }

    public get image(): HTMLImageElement | HTMLCanvasElement { return this.getImage(); }

    /**
     * Returns the current image's representation as either a Canvas or img element.
     *
     * NOTE: If a URL or mipmap data was provided, this currently doesn't return the original input to setImage(), but
     *       instead provides the mapped result (or first mipmap level's image).
     *       TODO: return the original result instead.
     */
    public getImage(): HTMLImageElement | HTMLCanvasElement {
      assert && assert( this._image !== null );

      return this._image!;
    }

    /**
     * Triggers recomputation of the image's bounds and refreshes any displays output of the image.
     *
     * Generally this can trigger recomputation of mipmaps, will mark any drawables as needing repaints, and will
     * cause a spritesheet change for WebGL.
     *
     * This should be done when the underlying image has changed appearance (usually the case with a Canvas changing,
     * but this is also triggered by our actual image reference changing).
     */
    public invalidateImage(): void {
      this.invalidateMipmaps();
      this._invalidateHitTestData();
    }

    /**
     * Sets the image with additional information about dimensions used before the image has loaded.
     *
     * This is essentially the same as setImage(), but also updates the initial dimensions. See setImage()'s
     * documentation for details on the image parameter.
     *
     * NOTE: setImage() will first reset the initial dimensions to 0, which will then be overridden later in this
     *       function. This may trigger bounds changes, even if the previous and next image (and image dimensions)
     *       are the same.
     *
     * @param image - See setImage()'s documentation
     * @param width - Initial width of the image. See setInitialWidth() for more documentation
     * @param height - Initial height of the image. See setInitialHeight() for more documentation
     */
    public setImageWithSize( image: string | HTMLImageElement | HTMLCanvasElement | Mipmap, width: number, height: number ): this {
      // First, setImage(), as it will reset the initial width and height
      this.setImage( image );

      // Then apply the initial dimensions
      this.setInitialWidth( width );
      this.setInitialHeight( height );

      return this;
    }

    /**
     * Sets an opacity that is applied only to this image (will not affect children or the rest of the node's subtree).
     *
     * This should generally be preferred over Node's opacity if it has the same result, as modifying this will be much
     * faster, and will not force additional Canvases or intermediate steps in display.
     *
     * @param imageOpacity - Should be a number between 0 (transparent) and 1 (opaque), just like normal
     *                                opacity.
     */
    public setImageOpacity( imageOpacity: number ): void {
      assert && assert( isFinite( imageOpacity ) && imageOpacity >= 0 && imageOpacity <= 1,
        `imageOpacity out of range: ${imageOpacity}` );

      if ( this._imageOpacity !== imageOpacity ) {
        this._imageOpacity = imageOpacity;
      }
    }

    public set imageOpacity( value: number ) { this.setImageOpacity( value ); }

    public get imageOpacity(): number { return this.getImageOpacity(); }

    /**
     * Returns the opacity applied only to this image (not including children).
     *
     * See setImageOpacity() documentation for more information.
     */
    public getImageOpacity(): number {
      return this._imageOpacity;
    }

    /**
     * Provides an initial width for an image that has not loaded yet.
     *
     * If the input image hasn't loaded yet, but the (expected) size is known, providing an initialWidth will cause the
     * Image node to have the correct bounds (width) before the pixel data has been fully loaded. A value of 0 will be
     * ignored.
     *
     * This is required for many browsers, as images can show up as a 0x0 (like Safari does for unloaded images).
     *
     * NOTE: setImage will reset this value to 0 (ignored), since it's potentially likely the new image has different
     *       dimensions than the current image.
     *
     * NOTE: If these dimensions end up being different than the actual image width/height once it has been loaded, an
     *       assertion will fail. Only the correct dimensions should be provided. If the width/height is unknown,
     *       please use the localBounds override or a transparent rectangle for taking up the (approximate) bounds.
     *
     * @param width - Expected width of the image's unloaded content
     */
    public setInitialWidth( width: number ): this {
      assert && assert( width >= 0 && ( width % 1 === 0 ), 'initialWidth should be a non-negative integer' );

      if ( width !== this._initialWidth ) {
        this._initialWidth = width;

        this.invalidateImage();
      }

      return this;
    }

    public set initialWidth( value: number ) { this.setInitialWidth( value ); }

    public get initialWidth(): number { return this.getInitialWidth(); }

    /**
     * Returns the initialWidth value set from setInitialWidth().
     *
     * See setInitialWidth() for more documentation. A value of 0 is ignored.
     */
    public getInitialWidth(): number {
      return this._initialWidth;
    }

    /**
     * Provides an initial height for an image that has not loaded yet.
     *
     * If the input image hasn't loaded yet, but the (expected) size is known, providing an initialWidth will cause the
     * Image node to have the correct bounds (height) before the pixel data has been fully loaded. A value of 0 will be
     * ignored.
     *
     * This is required for many browsers, as images can show up as a 0x0 (like Safari does for unloaded images).
     *
     * NOTE: setImage will reset this value to 0 (ignored), since it's potentially likely the new image has different
     *       dimensions than the current image.
     *
     * NOTE: If these dimensions end up being different than the actual image width/height once it has been loaded, an
     *       assertion will fail. Only the correct dimensions should be provided. If the width/height is unknown,
     *       please use the localBounds override or a transparent rectangle for taking up the (approximate) bounds.
     *
     * @param height - Expected height of the image's unloaded content
     */
    public setInitialHeight( height: number ): this {
      assert && assert( height >= 0 && ( height % 1 === 0 ), 'initialHeight should be a non-negative integer' );

      if ( height !== this._initialHeight ) {
        this._initialHeight = height;

        this.invalidateImage();
      }

      return this;
    }

    public set initialHeight( value: number ) { this.setInitialHeight( value ); }

    public get initialHeight(): number { return this.getInitialHeight(); }

    /**
     * Returns the initialHeight value set from setInitialHeight().
     *
     * See setInitialHeight() for more documentation. A value of 0 is ignored.
     */
    public getInitialHeight(): number {
      return this._initialHeight;
    }

    /**
     * Sets whether mipmapping is supported.
     *
     * This defaults to false, but is automatically set to true when a mipmap is provided to setImage(). Setting it to
     * true on non-mipmap images will trigger creation of a medium-quality mipmap that will be used.
     *
     * NOTE: This mipmap generation is slow and CPU-intensive. Providing precomputed mipmap resources to an Image node
     *       will be much faster, and of higher quality.
     *
     * @param mipmap - Whether mipmapping is supported
     */
    public setMipmap( mipmap: boolean ): this {
      if ( this._mipmap !== mipmap ) {
        this._mipmap = mipmap;

        this.invalidateMipmaps();
      }

      return this;
    }

    public set mipmap( value: boolean ) { this.setMipmap( value ); }

    public get mipmap(): boolean { return this.isMipmap(); }

    /**
     * Returns whether mipmapping is supported.
     *
     * See setMipmap() for more documentation.
     */
    public isMipmap(): boolean {
      return this._mipmap;
    }

    /**
     * Sets how much level-of-detail is displayed for mipmapping.
     *
     * When displaying mipmapped images as output, a certain source level of the mipmap needs to be used. Using a level
     * with too much resolution can create an aliased look (but will generally be sharper). Using a level with too
     * little resolution will be blurrier (but not aliased).
     *
     * The value of the mipmap bias is added on to the computed "ideal" mipmap level, and:
     * - A negative bias will typically increase the displayed resolution
     * - A positive bias will typically decrease the displayed resolution
     *
     * This is done approximately like the following formula:
     *   mipmapLevel = Utils.roundSymmetric( computedMipmapLevel + mipmapBias )
     */
    public setMipmapBias( bias: number ): this {
      if ( this._mipmapBias !== bias ) {
        this._mipmapBias = bias;

        this.invalidateMipmaps();
      }

      return this;
    }

    public set mipmapBias( value: number ) { this.setMipmapBias( value ); }

    public get mipmapBias(): number { return this.getMipmapBias(); }

    /**
     * Returns the current mipmap bias.
     *
     * See setMipmapBias() for more documentation.
     */
    public getMipmapBias(): number {
      return this._mipmapBias;
    }

    /**
     * The number of initial mipmap levels to compute (if Scenery generates the mipmaps by setting mipmap:true on a
     * non-mipmapped input).
     *
     * @param level - A non-negative integer representing the number of mipmap levels to precompute.
     */
    public setMipmapInitialLevel( level: number ): this {
      assert && assert( level % 1 === 0 && level >= 0,
        'mipmapInitialLevel should be a non-negative integer' );

      if ( this._mipmapInitialLevel !== level ) {
        this._mipmapInitialLevel = level;

        this.invalidateMipmaps();
      }

      return this;
    }

    public set mipmapInitialLevel( value: number ) { this.setMipmapInitialLevel( value ); }

    public get mipmapInitialLevel(): number { return this.getMipmapInitialLevel(); }

    /**
     * Returns the current initial mipmap level.
     *
     * See setMipmapInitialLevel() for more documentation.
     */
    public getMipmapInitialLevel(): number {
      return this._mipmapInitialLevel;
    }

    /**
     * The maximum (lowest-resolution) level that Scenery will compute if it generates mipmaps (e.g. by setting
     * mipmap:true on a non-mipmapped input).
     *
     * The default will precompute all default levels (from mipmapInitialLevel), so that we ideally don't hit mipmap
     * generation during animation.
     *
     * @param level - A non-negative integer representing the maximum mipmap level to compute.
     */
    public setMipmapMaxLevel( level: number ): this {
      assert && assert( level % 1 === 0 && level >= 0,
        'mipmapMaxLevel should be a non-negative integer' );

      if ( this._mipmapMaxLevel !== level ) {
        this._mipmapMaxLevel = level;

        this.invalidateMipmaps();
      }

      return this;
    }

    public set mipmapMaxLevel( value: number ) { this.setMipmapMaxLevel( value ); }

    public get mipmapMaxLevel(): number { return this.getMipmapMaxLevel(); }

    /**
     * Returns the current maximum mipmap level.
     *
     * See setMipmapMaxLevel() for more documentation.
     */
    public getMipmapMaxLevel(): number {
      return this._mipmapMaxLevel;
    }

    /**
     * Controls whether either any pixel in the image will be marked as contained (when false), or whether transparent
     * pixels will be counted as "not contained in the image" for hit-testing (when true).
     *
     * See https://github.com/phetsims/scenery/issues/1049 for more information.
     */
    public setHitTestPixels( hitTestPixels: boolean ): this {

      if ( this._hitTestPixels !== hitTestPixels ) {
        this._hitTestPixels = hitTestPixels;

        this._invalidateHitTestData();
      }

      return this;
    }

    public set hitTestPixels( value: boolean ) { this.setHitTestPixels( value ); }

    public get hitTestPixels(): boolean { return this.getHitTestPixels(); }

    /**
     * Returns whether pixels are checked for hit testing.
     *
     * See setHitTestPixels() for more documentation.
     */
    public getHitTestPixels(): boolean {
      return this._hitTestPixels;
    }

    /**
     * Constructs the next available (uncomputed) mipmap level, as long as the previous level was larger than 1x1.
     */
    private _constructNextMipmap(): void {
      const level = this._mipmapCanvases.length;
      const biggerCanvas = this._mipmapCanvases[ level - 1 ];

      // ignore any 1x1 canvases (or smaller?!?)
      if ( biggerCanvas.width * biggerCanvas.height > 2 ) {
        const canvas = document.createElement( 'canvas' );
        canvas.width = Math.ceil( biggerCanvas.width / 2 );
        canvas.height = Math.ceil( biggerCanvas.height / 2 );

        // sanity check
        if ( canvas.width > 0 && canvas.height > 0 ) {
          // Draw half-scale into the smaller Canvas
          const context = canvas.getContext( '2d' )!;
          context.scale( 0.5, 0.5 );
          context.drawImage( biggerCanvas, 0, 0 );

          this._mipmapCanvases.push( canvas );
          this._mipmapURLs.push( canvas.toDataURL() );
        }
      }
    }

    /**
     * Triggers recomputation of mipmaps (as long as mipmapping is enabled)
     */
    public invalidateMipmaps(): void {
      // Clean output arrays
      cleanArray( this._mipmapCanvases );
      cleanArray( this._mipmapURLs );

      if ( this._image && this._mipmap ) {
        // If we have mipmap data as an input
        if ( this._mipmapData ) {
          for ( let k = 0; k < this._mipmapData.length; k++ ) {
            const url = this._mipmapData[ k ].url;
            this._mipmapURLs.push( url );
            const updateCanvas = this._mipmapData[ k ].updateCanvas;
            updateCanvas && updateCanvas();
            this._mipmapCanvases.push( this._mipmapData[ k ].canvas! );
          }
        }
        // Otherwise, we have an image (not mipmap) as our input, so we'll need to construct mipmap levels.
        else {
          const baseCanvas = document.createElement( 'canvas' );
          baseCanvas.width = this.getImageWidth();
          baseCanvas.height = this.getImageHeight();

          // if we are not loaded yet, just ignore
          if ( baseCanvas.width && baseCanvas.height ) {
            const baseContext = baseCanvas.getContext( '2d' )!;
            baseContext.drawImage( this._image, 0, 0 );

            this._mipmapCanvases.push( baseCanvas );
            this._mipmapURLs.push( baseCanvas.toDataURL() );

            let level = 0;
            while ( ++level < this._mipmapInitialLevel ) {
              this._constructNextMipmap();
            }
          }
        }
      }

      this.mipmapEmitter.emit();
    }

    /**
     * Returns the desired mipmap level (0-indexed) that should be used for the particular relative transform. (scenery-internal)
     *
     * @param matrix - The relative transformation matrix of the node.
     * @param [additionalBias] - Can be provided to get per-call bias (we want some of this for Canvas output)
     */
    public getMipmapLevel( matrix: Matrix3, additionalBias = 0 ): number {
      assert && assert( this._mipmap, 'Assumes mipmaps can be used' );

      // Handle high-dpi devices like retina with correct mipmap levels.
      const scale = Imageable.getApproximateMatrixScale( matrix ) * ( window.devicePixelRatio || 1 );

      return this.getMipmapLevelFromScale( scale, additionalBias );
    }

    /**
     * Returns the desired mipmap level (0-indexed) that should be used for the particular scale
     */
    public getMipmapLevelFromScale( scale: number, additionalBias = 0 ): number {
      assert && assert( scale > 0, 'scale should be a positive number' );

      // If we are shown larger than scale, ALWAYS choose the highest resolution
      if ( scale >= 1 ) {
        return 0;
      }

      // our approximate level of detail
      let level = log2( 1 / scale );

      // convert to an integer level (-0.7 is a good default)
      level = Utils.roundSymmetric( level + this._mipmapBias + additionalBias - 0.7 );

      if ( level < 0 ) {
        level = 0;
      }
      if ( level > this._mipmapMaxLevel ) {
        level = this._mipmapMaxLevel;
      }

      // If necessary, do lazy construction of the mipmap level
      if ( this.mipmap && !this._mipmapCanvases[ level ] ) {
        let currentLevel = this._mipmapCanvases.length - 1;
        while ( ++currentLevel <= level ) {
          this._constructNextMipmap();
        }
        // Sanity check, since _constructNextMipmap() may have had to bail out. We had to compute some, so use the last
        return Math.min( level, this._mipmapCanvases.length - 1 );
      }
      // Should already be constructed, or isn't needed
      else {
        return level;
      }
    }

    /**
     * Returns a matching Canvas element for the given level-of-detail. (scenery-internal)
     *
     * @param level - Non-negative integer representing the mipmap level
     * @returns - Matching <canvas> for the level of detail
     */
    public getMipmapCanvas( level: number ): HTMLCanvasElement {
      assert && assert( level >= 0 &&
      level < this._mipmapCanvases.length &&
      ( level % 1 ) === 0 );

      // Sanity check to make sure we have copied the image data in if necessary.
      if ( this._mipmapData ) {
        // level may not exist (it was generated), and updateCanvas may not exist
        const updateCanvas = this._mipmapData[ level ] && this._mipmapData[ level ].updateCanvas;
        updateCanvas && updateCanvas();
      }
      return this._mipmapCanvases[ level ];
    }

    /**
     * Returns a matching URL string for an image for the given level-of-detail. (scenery-internal)
     *
     * @param level - Non-negative integer representing the mipmap level
     * @returns - Matching data URL for the level of detail
     */
    public getMipmapURL( level: number ): string {
      assert && assert( level >= 0 &&
      level < this._mipmapCanvases.length &&
      ( level % 1 ) === 0 );

      return this._mipmapURLs[ level ];
    }

    /**
     * Returns whether there are mipmap levels that have been computed. (scenery-internal)
     */
    public hasMipmaps(): boolean {
      return this._mipmapCanvases.length > 0;
    }

    /**
     * Triggers recomputation of hit test data
     */
    private _invalidateHitTestData(): void {
      // Only compute this if we are hit-testing pixels
      if ( !this._hitTestPixels ) {
        return;
      }

      if ( this._image !== null ) {
        this._hitTestImageData = Imageable.getHitTestData( this._image, this.imageWidth, this.imageHeight );
      }
    }

    /**
     * Returns the width of the displayed image (not related to how this node is transformed).
     *
     * NOTE: If the image is not loaded and an initialWidth was provided, that width will be used.
     */
    public getImageWidth(): number {
      if ( this._image === null ) {
        return 0;
      }

      const detectedWidth = this._mipmapData ? this._mipmapData[ 0 ].width : ( ( 'naturalWidth' in this._image ? this._image.naturalWidth : 0 ) || this._image.width );
      if ( detectedWidth === 0 ) {
        return this._initialWidth; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialWidth === 0 || this._initialWidth === detectedWidth, 'Bad Image.initialWidth' );

        return detectedWidth;
      }
    }

    public get imageWidth(): number { return this.getImageWidth(); }

    /**
     * Returns the height of the displayed image (not related to how this node is transformed).
     *
     * NOTE: If the image is not loaded and an initialHeight was provided, that height will be used.
     */
    public getImageHeight(): number {
      if ( this._image === null ) {
        return 0;
      }

      const detectedHeight = this._mipmapData ? this._mipmapData[ 0 ].height : ( ( 'naturalHeight' in this._image ? this._image.naturalHeight : 0 ) || this._image.height );
      if ( detectedHeight === 0 ) {
        return this._initialHeight; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialHeight === 0 || this._initialHeight === detectedHeight, 'Bad Image.initialHeight' );

        return detectedHeight;
      }
    }

    public get imageHeight(): number { return this.getImageHeight(); }

    /**
     * If our provided image is an HTMLImageElement, returns its URL (src). (scenery-internal)
     */
    public getImageURL(): string {
      assert && assert( this._image instanceof HTMLImageElement, 'Only supported for HTML image elements' );

      return ( this._image as HTMLImageElement ).src;
    }

    /**
     * Attaches our on-load listener to our current image.
     */
    private _attachImageLoadListener(): void {
      assert && assert( !this._imageLoadListenerAttached, 'Should only be attached to one thing at a time' );

      if ( !this.isDisposed ) {
        ( this._image as HTMLImageElement ).addEventListener( 'load', this._imageLoadListener );
        this._imageLoadListenerAttached = true;
      }
    }

    /**
     * Detaches our on-load listener from our current image.
     */
    private _detachImageLoadListener(): void {
      assert && assert( this._imageLoadListenerAttached, 'Needs to be attached first to be detached.' );

      ( this._image as HTMLImageElement ).removeEventListener( 'load', this._imageLoadListener );
      this._imageLoadListenerAttached = false;
    }

    /**
     * Called when our image has loaded (it was not yet loaded with then listener was added)
     */
    private _onImageLoad(): void {
      assert && assert( this._imageLoadListenerAttached, 'If _onImageLoad is firing, it should be attached' );

      this.invalidateImage();
      this._detachImageLoadListener();
    }

    /**
     * Disposes the path, releasing image listeners if needed (and preventing new listeners from being added).
     */
    public dispose(): void {
      if ( this._image && this._imageLoadListenerAttached ) {
        this._detachImageLoadListener();
      }

      // @ts-expect-error
      super.dispose && super.dispose();
    }
  };
};

/**
 * Optionally returns an ImageData object useful for hit-testing the pixel data of an image.
 *
 * @param image
 * @param width - logical width of the image
 * @param height - logical height of the image
 */
Imageable.getHitTestData = ( image: HTMLImageElement | HTMLCanvasElement, width: number, height: number ): ImageData | null => {
  // If the image isn't loaded yet, we don't want to try loading anything
  if ( !( ( 'naturalWidth' in image ? image.naturalWidth : 0 ) || image.width ) || !( ( 'naturalHeight' in image ? image.naturalHeight : 0 ) || image.height ) ) {
    return null;
  }

  const canvas = getScratchCanvas();
  const context = getScratchContext();

  canvas.width = width;
  canvas.height = height;
  context.drawImage( image, 0, 0 );

  return context.getImageData( 0, 0, width, height );
};

/**
 * Tests whether a given pixel in an ImageData is at all non-transparent.
 *
 * @param imageData
 * @param width - logical width of the image
 * @param height - logical height of the image
 * @param point
 */
Imageable.testHitTestData = ( imageData: ImageData, width: number, height: number, point: Vector2 ): boolean => {
  // For sanity, map it based on the image dimensions and image data dimensions, and carefully clamp in case things are weird.
  const x = Utils.clamp( Math.floor( ( point.x / width ) * imageData.width ), 0, imageData.width - 1 );
  const y = Utils.clamp( Math.floor( ( point.y / height ) * imageData.height ), 0, imageData.height - 1 );

  const index = 4 * ( x + y * imageData.width ) + 3;

  return imageData.data[ index ] !== 0;
};

/**
 * Turns the ImageData into a Shape showing where hit testing would succeed.
 *
 * @param imageData
 * @param width - logical width of the image
 * @param height - logical height of the image
 */
Imageable.hitTestDataToShape = ( imageData: ImageData, width: number, height: number ): Shape => {
  const widthScale = width / imageData.width;
  const heightScale = height / imageData.height;

  const shape = new Shape();

  // Create rows at a time, so that if we have 50 adjacent pixels "on", then we'll just make a rectangle 50-wide.
  // This lets us do the CAG faster.
  let active = false;
  let min = 0;

  // NOTE: Rows are more helpful for CAG, even though columns would have better cache behavior when accessing the
  // imageData.

  for ( let y = 0; y < imageData.height; y++ ) {
    for ( let x = 0; x < imageData.width; x++ ) {
      const index = 4 * ( x + y * imageData.width ) + 3;

      if ( imageData.data[ index ] !== 0 ) {
        // If our last pixel was empty, and now we're "on", start our rectangle
        if ( !active ) {
          active = true;
          min = x;
        }
      }
      else if ( active ) {
        // Finish a rectangle once we reach an "off" pixel
        active = false;
        shape.rect( min * widthScale, y * widthScale, widthScale * ( x - min ), heightScale );
      }
    }
    if ( active ) {
      // We'll need to finish rectangles at the end of each row anyway.
      active = false;
      shape.rect( min * widthScale, y * widthScale, widthScale * ( imageData.width - min ), heightScale );
    }
  }

  return shape.getSimplifiedAreaShape();
};

/**
 * Creates an SVG image element with a given URL and dimensions
 *
 * @param url - The URL for the image
 * @param width - Non-negative integer for the image's width
 * @param height - Non-negative integer for the image's height
 */
Imageable.createSVGImage = ( url: string, width: number, height: number ): SVGImageElement => {
  assert && assert( isFinite( width ) && width >= 0 && ( width % 1 ) === 0,
    'width should be a non-negative finite integer' );
  assert && assert( isFinite( height ) && height >= 0 && ( height % 1 ) === 0,
    'height should be a non-negative finite integer' );

  const element = document.createElementNS( svgns, 'image' );
  element.setAttribute( 'x', '0' );
  element.setAttribute( 'y', '0' );
  element.setAttribute( 'width', `${width}px` );
  element.setAttribute( 'height', `${height}px` );
  element.setAttributeNS( xlinkns, 'xlink:href', url );

  return element;
};

/**
 * Creates an object suitable to be passed to Image as a mipmap (from a Canvas)
 */
Imageable.createFastMipmapFromCanvas = ( baseCanvas: HTMLCanvasElement ): Mipmap => {
  const mipmaps: Mipmap = [];

  const baseURL = baseCanvas.toDataURL();
  const baseImage = new window.Image();
  baseImage.src = baseURL;

  // base level
  mipmaps.push( {
    img: baseImage,
    url: baseURL,
    width: baseCanvas.width,
    height: baseCanvas.height,
    canvas: baseCanvas
  } );

  let largeCanvas = baseCanvas;
  while ( largeCanvas.width >= 2 && largeCanvas.height >= 2 ) {

    // draw half-size
    const canvas = document.createElement( 'canvas' );
    canvas.width = Math.ceil( largeCanvas.width / 2 );
    canvas.height = Math.ceil( largeCanvas.height / 2 );
    const context = canvas.getContext( '2d' )!;
    context.setTransform( 0.5, 0, 0, 0.5, 0, 0 );
    context.drawImage( largeCanvas, 0, 0 );

    // smaller level
    const mipmapLevel = {
      width: canvas.width,
      height: canvas.height,
      canvas: canvas,
      url: canvas.toDataURL(),
      img: new window.Image()
    };
    // set up the image and url
    mipmapLevel.img.src = mipmapLevel.url;

    largeCanvas = canvas;
    mipmaps.push( mipmapLevel );
  }

  return mipmaps;
};

/**
 * Returns a sense of "average" scale, which should be exact if there is no asymmetric scale/shear applied
 */
Imageable.getApproximateMatrixScale = ( matrix: Matrix3 ): number => {
  return ( Math.sqrt( matrix.m00() * matrix.m00() + matrix.m10() * matrix.m10() ) +
           Math.sqrt( matrix.m01() * matrix.m01() + matrix.m11() * matrix.m11() ) ) / 2;
};

// {number} - We include this for additional smoothing that seems to be needed for Canvas image quality
Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT = 0.5;

// {Object} - Initial values for most Node mutator options
Imageable.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'Imageable', Imageable );
export default Imageable;
