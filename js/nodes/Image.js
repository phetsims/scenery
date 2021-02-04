// Copyright 2013-2020, University of Colorado Boulder

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
import ImageCanvasDrawable from '../display/drawables/ImageCanvasDrawable.js';
import ImageDOMDrawable from '../display/drawables/ImageDOMDrawable.js';
import ImageSVGDrawable from '../display/drawables/ImageSVGDrawable.js';
import ImageWebGLDrawable from '../display/drawables/ImageWebGLDrawable.js';
import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';
import SpriteSheet from '../util/SpriteSheet.js';
import Imageable from './Imageable.js';
import Node from './Node.js';

// Image-specific options that can be passed in the constructor or mutate() call.
const IMAGE_OPTION_KEYS = [
  'image', // {string|HTMLImageElement|HTMLCanvasElement|Array} - Changes the image displayed, see setImage() for documentation
  'imageOpacity', // {number} - Controls opacity of this image (and not children), see setImageOpacity() for documentation
  'initialWidth', // {number} - Width of an image not-yet loaded (for layout), see setInitialWidth() for documentation
  'initialHeight', // {number} - Height of an image not-yet loaded (for layout), see setInitialHeight() for documentation
  'mipmap', // {boolean} - Whether mipmapped output is supported, see setMipmap() for documentation
  'mipmapBias', // {number} - Whether mipmapping tends towards sharp/aliased or blurry, see setMipmapBias() for documentation
  'mipmapInitialLevel', // {number} - How many mipmap levels to generate if needed, see setMipmapInitialLevel() for documentation
  'mipmapMaxLevel', // {number} The maximum mipmap level to compute if needed, see setMipmapMaxLevel() for documentation
  'hitTestPixels' // {boolean} - Whether non-transparent pixels will control contained points, see setHitTestPixels() for documentation
];

class Image extends Imageable( Node ) {
  /**
   * Constructs an Image node from a particular source.
   *
   * IMAGE_OPTION_KEYS (above) describes the available options keys that can be provided, on top of Node's options.
   *
   * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image - See setImage() for details.
   * @param {Object} [options] - Image-specific options are documented in IMAGE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  constructor( image, options ) {

    super();

    // rely on the setImage call from the super constructor to do the setup
    options = extendDefined( {
      image: image
    }, options );

    this.mutate( options );

    this.invalidateSupportedRenderers();
  }

  /**
   * Triggers recomputation of the image's bounds and refreshes any displays output of the image.
   * @public
   * @override
   *
   * Generally this can trigger recomputation of mipmaps, will mark any drawables as needing repaints, and will
   * cause a spritesheet change for WebGL.
   *
   * This should be done when the underlying image has changed appearance (usually the case with a Canvas changing,
   * but this is also triggered by our actual image reference changing).
   */
  invalidateImage() {
    if ( this._image ) {
      this.invalidateSelf( new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
    }
    else {
      this.invalidateSelf( Bounds2.NOTHING );
    }

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      this._drawables[ i ].markDirtyImage();
    }

    super.invalidateImage();

    this.invalidateSupportedRenderers();
  }

  /**
   * Recomputes what renderers are supported, given the current image information.
   * @private
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
   * @public
   * @override
   *
   * This should generally be preferred over Node's opacity if it has the same result, as modifying this will be much
   * faster, and will not force additional Canvases or intermediate steps in display.
   *
   * @param {number} imageOpacity - Should be a number between 0 (transparent) and 1 (opaque), just like normal
   *                                opacity.
   */
  setImageOpacity( imageOpacity ) {
    const changed = this._imageOpacity !== imageOpacity;

    super.setImageOpacity( imageOpacity );

    if ( changed ) {
      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyImageOpacity();
      }
    }
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   * @public
   * @override
   *
   * @returns {boolean}
   */
  isPainted() {
    // Always true for Image nodes
    return true;
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   * @protected
   * @override
   *
   * @param {CanvasContextWrapper} wrapper
   * @param {Matrix3} matrix - The transformation matrix already applied to the context.
   */
  canvasPaintSelf( wrapper, matrix ) {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    ImageCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Image.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {DOMSelfDrawable}
   */
  createDOMDrawable( renderer, instance ) {
    return ImageDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Image.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {SVGSelfDrawable}
   */
  createSVGDrawable( renderer, instance ) {
    return ImageSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Image.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {CanvasSelfDrawable}
   */
  createCanvasDrawable( renderer, instance ) {
    return ImageCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a WebGL drawable for this Image.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {WebGLSelfDrawable}
   */
  createWebGLDrawable( renderer, instance ) {
    return ImageWebGLDrawable.createFromPool( renderer, instance );
  }

  /**
   * Override this for computation of whether a point is inside our self content (defaults to selfBounds check).
   * @protected
   * @override
   *
   * @param {Vector2} point - Considered to be in the local coordinate frame
   * @returns {boolean}
   */
  containsPointSelf( point ) {
    const inBounds = Node.prototype.containsPointSelf.call( this, point );

    if ( !inBounds || !this._hitTestPixels || !this._hitTestImageData ) {
      return inBounds;
    }

    return Imageable.testHitTestData( this._hitTestImageData, this.imageWidth, this.imageHeight, point );
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   * @public
   *
   * @returns {Shape}
   */
  getSelfShape() {
    if ( this._hitTestPixels ) {
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
   * @protected
   * @override
   */
  invalidateMipmaps() {
    const markDirty = this._image && this._mipmap && !this._mipmapData;

    super.invalidateMipmaps();

    if ( markDirty ) {
      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyMipmap();
      }
    }
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Image.prototype._mutatorKeys = IMAGE_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * @public (scenery-internal)
 * @override
 */
Image.prototype.drawableMarkFlags = Node.prototype.drawableMarkFlags.concat( [ 'image', 'imageOpacity', 'mipmap' ] );

// @public {Object} - Initial values for most Node mutator options
Image.DEFAULT_OPTIONS = merge( {}, Node.DEFAULT_OPTIONS, Imageable.DEFAULT_OPTIONS );

// NOTE: Not currently in use
Image.IOType = new IOType( 'ImageIO', {
  valueType: Image,
  supertype: Node.NodeIO,
  events: [ 'changed' ],
  methods: {
    setImage: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( base64Text ) {
        const im = new window.Image();
        im.src = base64Text;
        this.image = im;
      },
      documentation: 'Set the image from a base64 string',
      invocableForReadOnlyElements: false
    }
  }
} );

scenery.register( 'Image', Image );
export default Image;