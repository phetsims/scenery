// Copyright 2013-2021, University of Colorado Boulder

/**
 * A node that displays a single image either from an actual HTMLImageElement, a URL, a Canvas element, or a mipmap
 * data structure described in the constructor.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import merge from '../../../phet-core/js/merge.js';
import IOType from '../../../tandem/js/types/IOType.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Shape from '../../../kite/js/Shape.js';
import { scenery, SpriteSheet, Imageable, Renderer, Node, NodeOptions, CanvasContextWrapper, ImageCanvasDrawable, ImageDOMDrawable, ImageSVGDrawable, ImageWebGLDrawable, Instance, DOMSelfDrawable, SVGSelfDrawable, CanvasSelfDrawable, WebGLSelfDrawable, IImageDrawable } from '../imports.js';

// Image-specific options that can be passed in the constructor or mutate() call.
const IMAGE_OPTION_KEYS = [
  'image', // {string|HTMLImageElement|HTMLCanvasElement|Array} - Changes the image displayed, see setImage() for documentation
  'imageOpacity', // {number} - Controls opacity of this image (and not children), see setImageOpacity() for documentation
  'imageBounds', // {Bounds2|null} - Controls the amount of the image that is hit-tested or considered "inside" the image, see setImageBounds()
  'initialWidth', // {number} - Width of an image not-yet loaded (for layout), see setInitialWidth() for documentation
  'initialHeight', // {number} - Height of an image not-yet loaded (for layout), see setInitialHeight() for documentation
  'mipmap', // {boolean} - Whether mipmapped output is supported, see setMipmap() for documentation
  'mipmapBias', // {number} - Whether mipmapping tends towards sharp/aliased or blurry, see setMipmapBias() for documentation
  'mipmapInitialLevel', // {number} - How many mipmap levels to generate if needed, see setMipmapInitialLevel() for documentation
  'mipmapMaxLevel', // {number} The maximum mipmap level to compute if needed, see setMipmapMaxLevel() for documentation
  'hitTestPixels' // {boolean} - Whether non-transparent pixels will control contained points, see setHitTestPixels() for documentation
];

type Mipmap = {
  width: number,
  height: number,
  url: string,
  canvas?: HTMLCanvasElement,
  img?: HTMLImageElement,
  updateCanvas?: () => void
}[];

type ImageOptions = {
  image?: string | HTMLImageElement | HTMLCanvasElement | Mipmap,
  imageOpacity?: number,
  imageBounds?: Bounds2 | null,
  initialWidth?: number,
  initialHeight?: number,
  mipmap?: boolean,
  mipmapBias?: number,
  mipmapInitialLevel?: number,
  mipmapMaxLevel?: number,
  hitTestPixels?: boolean
} & NodeOptions;

/**
 * @extends Node
 */
class Image extends Imageable( Node ) {

  // If non-null, determines what is considered "inside" the image for containment and hit-testing.
  _imageBounds: Bounds2 | null;

  constructor( image: string | HTMLImageElement | HTMLCanvasElement | Mipmap, options?: ImageOptions ) {

    super();

    // rely on the setImage call from the super constructor to do the setup
    options = extendDefined( {
      image: image
    }, options );

    this._imageBounds = null;

    this.mutate( options );

    this.invalidateSupportedRenderers();
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
  invalidateImage() {
    if ( this._image ) {
      this.invalidateSelf( this._imageBounds || new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
    }
    else {
      this.invalidateSelf( Bounds2.NOTHING );
    }

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      ( this._drawables[ i ] as unknown as IImageDrawable ).markDirtyImage();
    }

    super.invalidateImage();

    this.invalidateSupportedRenderers();
  }

  /**
   * Recomputes what renderers are supported, given the current image information.
   */
  invalidateSupportedRenderers() {

    // Canvas is always permitted
    let r = Renderer.bitmaskCanvas;

    // If it fits within the sprite sheet, then WebGL is also permitted
    // If the image hasn't loaded, the getImageWidth/Height will be 0 and this rule would pass.  However, this
    // function will be called again after the image loads, and would correctly invalidate WebGL, if too large to fit
    // in a SpriteSheet
    const fitsWithinSpriteSheet = this.getImageWidth() <= SpriteSheet.MAX_DIMENSION.width &&
                                  this.getImageHeight() <= SpriteSheet.MAX_DIMENSION.height;
    if ( fitsWithinSpriteSheet ) {
      r |= Renderer.bitmaskWebGL;
    }

    // If it is not a canvas, then it can additionally be rendered in SVG or DOM
    if ( !( this._image instanceof HTMLCanvasElement ) ) {
      // assumes HTMLImageElement
      r |= Renderer.bitmaskSVG | Renderer.bitmaskDOM;
    }

    this.setRendererBitmask( r );
  }

  /**
   * Sets an opacity that is applied only to this image (will not affect children or the rest of the node's subtree).
   *
   * This should generally be preferred over Node's opacity if it has the same result, as modifying this will be much
   * faster, and will not force additional Canvases or intermediate steps in display.
   *
   * @param imageOpacity - Should be a number between 0 (transparent) and 1 (opaque), just like normal opacity.
   */
  setImageOpacity( imageOpacity: number ) {
    const changed = this._imageOpacity !== imageOpacity;

    super.setImageOpacity( imageOpacity );

    if ( changed ) {
      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IImageDrawable ).markDirtyImageOpacity();
      }
    }
  }

  /**
   * Sets the imageBounds value for the Image. If non-null, determines what is considered "inside" the image for
   * containment and hit-testing.
   *
   * NOTE: This is accomplished by using any provided imageBounds as the node's own selfBounds. This will affect layout,
   * hit-testing, and anything else using the bounds of this node.
   */
  setImageBounds( imageBounds: Bounds2 | null ) {
    if ( this._imageBounds !== imageBounds ) {
      this._imageBounds = imageBounds;

      this.invalidateImage();
    }
  }

  set imageBounds( value: Bounds2 | null ) { this.setImageBounds( value ); }

  /**
   * Returns the imageBounds, see setImageBounds for details.
   */
  getImageBounds(): Bounds2 | null {
    return this._imageBounds;
  }

  get imageBounds(): Bounds2 | null { return this._imageBounds; }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  isPainted() {
    // Always true for Image nodes
    return true;
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ) {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    ImageCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Image. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    // @ts-ignore - Poolable
    return ImageDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Image. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-ignore - Poolable
    return ImageSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Image. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   * @returns {CanvasSelfDrawable}
   */
  createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-ignore - Poolable
    return ImageCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a WebGL drawable for this Image. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   * @returns {WebGLSelfDrawable}
   */
  createWebGLDrawable( renderer: number, instance: Instance ): WebGLSelfDrawable {
    // @ts-ignore - Poolable
    return ImageWebGLDrawable.createFromPool( renderer, instance );
  }

  /**
   * Override this for computation of whether a point is inside our self content (defaults to selfBounds check).
   *
   * @param point - Considered to be in the local coordinate frame
   */
  containsPointSelf( point: Vector2 ): boolean {
    const inBounds = Node.prototype.containsPointSelf.call( this, point );

    if ( !inBounds || !this._hitTestPixels || !this._hitTestImageData ) {
      return inBounds;
    }

    return Imageable.testHitTestData( this._hitTestImageData, this.imageWidth, this.imageHeight, point );
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  getSelfShape(): Shape {
    if ( this._hitTestPixels && this._hitTestImageData ) {
      // If we're hit-testing pixels, return that shape included.
      return Imageable.hitTestDataToShape( this._hitTestImageData, this.imageWidth, this.imageHeight );
    }
    else {
      // Otherwise the super call will just include the rectangle (bounds).
      return Node.prototype.getSelfShape.call( this );
    }
  }

  /**
   * Triggers recomputation of mipmaps (as long as mipmapping is enabled)
   */
  invalidateMipmaps() {
    const markDirty = this._image && this._mipmap && !this._mipmapData;

    super.invalidateMipmaps();

    if ( markDirty ) {
      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IImageDrawable ).markDirtyMipmap();
      }
    }
  }

  static ImageIO: IOType;
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @public
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Image.prototype._mutatorKeys = [ ...IMAGE_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * @public (scenery-internal)
 * @override
 */
Image.prototype.drawableMarkFlags = [ ...Node.prototype.drawableMarkFlags, 'image', 'imageOpacity', 'mipmap' ];

// @public {Object} - Initial values for most Node mutator options
Image.DEFAULT_OPTIONS = merge( {}, Node.DEFAULT_OPTIONS, Imageable.DEFAULT_OPTIONS );

// NOTE: Not currently in use
Image.ImageIO = new IOType( 'ImageIO', {
  valueType: Image,
  supertype: Node.NodeIO,
  events: [ 'changed' ],
  methods: {
    setImage: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( base64Text: string ) {
        const im = new window.Image();
        im.src = base64Text;
        // @ts-ignore TODO: how would this even work?
        this.image = im;
      },
      documentation: 'Set the image from a base64 string',
      invocableForReadOnlyElements: false
    }
  }
} );

scenery.register( 'Image', Image );
export { Image as default };
export type { ImageOptions, Mipmap };
