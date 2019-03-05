// Copyright 2013-2016, University of Colorado Boulder

/**
 * A node that displays a single image either from an actual HTMLImageElement, a URL, a Canvas element, or a mipmap
 * data structure described in the constructor.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var ImageCanvasDrawable = require( 'SCENERY/display/drawables/ImageCanvasDrawable' );
  var ImageDOMDrawable = require( 'SCENERY/display/drawables/ImageDOMDrawable' );
  var ImageSVGDrawable = require( 'SCENERY/display/drawables/ImageSVGDrawable' );
  var ImageWebGLDrawable = require( 'SCENERY/display/drawables/ImageWebGLDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var SpriteSheet = require( 'SCENERY/util/SpriteSheet' );
  var Util = require( 'DOT/Util' );

  // Need to poly-fill on some browsers
  var log2 = Math.log2 || function( x ) { return Math.log( x ) / Math.LN2; };

  // Image-specific options that can be passed in the constructor or mutate() call.
  var IMAGE_OPTION_KEYS = [
    'image', // Changes the image displayed, see setImage() for documentation
    'imageOpacity', // Controls opacity of this image (and not children), see setImageOpacity() for documentation
    'initialWidth', // Width of an image not-yet loaded (for layout), see setInitialWidth() for documentation
    'initialHeight', // Height of an image not-yet loaded (for layout), see setInitialHeight() for documentation
    'mipmap', // Whether mipmapped output is supported, see setMipmap() for documentation
    'mipmapBias', // Whether mipmapping tends towards sharp/aliased or blurry, see setMipmapBias() for documentation
    'mipmapInitialLevel', // How many mipmap levels to generate if needed, see setMipmapInitialLevel() for documentation
    'mipmapMaxLevel' // The maximum mipmap level to compute if needed, see setMipmapMaxLevel() for documentation
  ];

  var DEFAULT_OPTIONS = {
    imageOpacity: 1,
    initialWidth: 0,
    initialHeight: 0,
    mipmap: false,
    mipmapBias: 0,
    mipmapInitialLevel: 4,
    mipmapMaxLevel: 5
  };

  /**
   * Constructs an Image node from a particular source.
   * @public
   * @constructor
   * @extends Node
   *
   * IMAGE_OPTION_KEYS (above) describes the available options keys that can be provided, on top of Node's options.
   *
   * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image - See setImage() for details.
   * @param {Object} [options] - Image-specific options are documented in IMAGE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  function Image( image, options ) {
    assert && assert( image, 'image should be available' );
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    // @private {number} - Internal stateful value, see setInitialWidth() for documentation.
    this._initialWidth = DEFAULT_OPTIONS.initialWidth;

    // @private {number} - Internal stateful value, see setInitialHeight() for documentation.
    this._initialHeight = DEFAULT_OPTIONS.initialHeight;

    // @private {number} - Internal stateful value, see setImageOpacity() for documentation.
    this._imageOpacity = DEFAULT_OPTIONS.imageOpacity;

    // @private {boolean} - Internal stateful value, see setMipmap() for documentation.
    this._mipmap = DEFAULT_OPTIONS.mipmap;

    // @private {number} - Internal stateful value, see setMipmapBias() for documentation.
    this._mipmapBias = DEFAULT_OPTIONS.mipmapBias;

    // @private {number} - Internal stateful value, see setMipmapInitialLevel() for documentation.
    this._mipmapInitialLevel = DEFAULT_OPTIONS.mipmapInitialLevel;

    // @private {number} - Internal stateful value, see setMipmapMaxLevel() for documentation
    this._mipmapMaxLevel = DEFAULT_OPTIONS.mipmapMaxLevel;

    // @private {Array.<HTMLCanvasElement>} - Array of Canvases for each level, constructed internally so that
    //                                        Canvas-based drawables (Canvas, WebGL) can quickly draw mipmaps.
    this._mipmapCanvases = [];

    // @private {Array.<String>} - Array of URLs for each level, where each URL will display an image (and is typically
    //                             a data URI or blob URI), so that we can handle mipmaps in SVG where URLs are
    //                             required.
    this._mipmapURLs = [];

    // @private {Array|null} - Mipmap data if it is passed into our image. Will be stored here for processing
    this._mipmapData = null;

    // @private {function} - Listener for invalidating our bounds whenever an image is invalidated.
    this._imageLoadListener = this.onImageLoad.bind( this );

    // @private {boolean} - Whether our _imageLoadListener has been attached as a listener to the current image.
    this._imageLoadListenerAttached = false;

    // rely on the setImage call from the super constructor to do the setup
    options = extendDefined( {
      image: image
    }, options );

    Node.call( this, options );

    this.invalidateSupportedRenderers();
  }

  scenery.register( 'Image', Image );

  inherit( Node, Image, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: IMAGE_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Node.prototype.drawableMarkFlags.concat( [ 'image', 'imageOpacity', 'mipmap' ] ),

    /**
     * Sets the current image to be displayed by this Image node.
     * @public
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
     *
     * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image - See documentation above
     * @returns {Image} - Self reference for chaining
     */
    setImage: function( image ) {
      assert && assert( image, 'image should be available' );
      assert && assert( typeof image === 'string' ||
                        image instanceof HTMLImageElement ||
                        image instanceof HTMLCanvasElement ||
                        Array.isArray( image ), 'image is not of the correct type' );

      // Generally, if a different value for image is provided, it has changed
      var hasImageChanged = this._image !== image;

      // Except in some cases, where the provided image is a string
      if ( hasImageChanged && typeof image === 'string' ) {

        // If our current image has the same .src as the "new" image, it's basically the same (as we promote string
        // images to HTMLImageElements).
        if ( this._image && image === this._image.src ) {
          hasImageChanged = false;
        }

        // If our current mipmap data is the same as the input, then we aren't changing it
        if ( image === this._mipmapData ) {
          hasImageChanged = false;
        }
      }

      if ( hasImageChanged ) {
        // Reset the initial dimensions, since we have a new image that may have different dimensions.
        this._initialWidth = 0;
        this._initialHeight = 0;

        // Don't leak memory by referencing old images
        if ( this._image && this._imageLoadListenerAttached ) {
          this.detachImageLoadListener();
        }

        // clear old mipmap data references
        this._mipmapData = null;

        // Convert string => HTMLImageElement
        if ( typeof image === 'string' ) {
          // create an image with the assumed URL
          var src = image;
          image = document.createElement( 'img' );
          image.src = src;
        }
        // Handle the provided mipmap
        else if ( Array.isArray( image ) ) {
          // mipmap data!
          this._mipmapData = image;
          image = image[ 0 ].img; // presumes we are already loaded

          // force initialization of mipmapping parameters, since invalidateMipmaps() is guaranteed to run below
          this._mipmapInitialLevel = this._mipmapMaxLevel = this._mipmapData.length;
          this._mipmap = true;
        }

        this._image = image;

        // If our image is an HTML image that hasn't loaded yet, attach a load listener.
        if ( this._image instanceof HTMLImageElement && ( !this._image.width || !this._image.height ) ) {
          this.attachImageLoadListener();
        }

        // Try recomputing bounds (may give a 0x0 if we aren't yet loaded)
        this.invalidateImage();
      }
      return this;
    },
    set image( value ) { this.setImage( value ); },

    /**
     * Returns the current image's representation as either a Canvas or img element.
     * @public
     *
     * NOTE: If a URL or mipmap data was provided, this currently doesn't return the original input to setImage(), but
     *       instead provides the mapped result (or first mipmap level's image).
     *       TODO: return the original result instead.
     *
     * @returns {HTMLImageElement|HTMLCanvasElement}
     */
    getImage: function() {
      return this._image;
    },
    get image() { return this.getImage(); },

    /**
     * Triggers recomputation of the image's bounds and refreshes any displays output of the image.
     * @public
     *
     * Generally this can trigger recomputation of mipmaps, will mark any drawables as needing repaints, and will
     * cause a spritesheet change for WebGL.
     *
     * This should be done when the underlying image has changed appearance (usually the case with a Canvas changing,
     * but this is also triggered by our actual image reference changing).
     */
    invalidateImage: function() {
      if ( this._image ) {
        this.invalidateSelf( new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
      }
      else {
        this.invalidateSelf( Bounds2.NOTHING );
      }

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyImage();
      }

      this.invalidateMipmaps();
      this.invalidateSupportedRenderers();
    },

    /**
     * Recomputes what renderers are supported, given the current image information.
     * @private
     */
    invalidateSupportedRenderers: function() {

      // Canvas is always permitted
      var r = Renderer.bitmaskCanvas;

      // If it fits within the sprite sheet, then WebGL is also permitted
      // If the image hasn't loaded, the getImageWidth/Height will be 0 and this rule would pass.  However, this
      // function will be called again after the image loads, and would correctly invalidate WebGL, if too large to fit
      // in a SpriteSheet
      var fitsWithinSpriteSheet = this.getImageWidth() <= SpriteSheet.MAX_DIMENSION.width &&
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
    },

    /**
     * Sets the image with additional information about dimensions used before the image has loaded.
     * @public
     *
     * This is essentially the same as setImage(), but also updates the initial dimensions. See setImage()'s
     * documentation for details on the image parameter.
     *
     * NOTE: setImage() will first reset the initial dimensions to 0, which will then be overridden later in this
     *       function. This may trigger bounds changes, even if the previous and next image (and image dimensions)
     *       are the same.
     *
     * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image - See setImage()'s documentation
     * @param {number} width - Initial width of the image. See setInitialWidth() for more documentation
     * @param {number} height - Initial height of the image. See setInitialHeight() for more documentation
     * @returns {Image} - For chaining
     */
    setImageWithSize: function( image, width, height ) {
      // First, setImage(), as it will reset the initial width and height
      this.setImage( image );

      // Then apply the initial dimensions
      this.setInitialWidth( width );
      this.setInitialHeight( height );

      return this;
    },

    /**
     * Sets an opacity that is applied only to this image (will not affect children or the rest of the node's subtree).
     * @public
     *
     * This should generally be preferred over Node's opacity if it has the same result, as modifying this will be much
     * faster, and will not force additional Canvases or intermediate steps in display.
     *
     * @param {number} imageOpacity - Should be a number between 0 (transparent) and 1 (opaque), just like normal
     *                                opacity.
     */
    setImageOpacity: function( imageOpacity ) {
      assert && assert( typeof imageOpacity === 'number', 'imageOpacity was not a number' );
      assert && assert( isFinite( imageOpacity ) && imageOpacity >= 0 && imageOpacity <= 1,
        'imageOpacity out of range: ' + imageOpacity );

      if ( this._imageOpacity !== imageOpacity ) {
        this._imageOpacity = imageOpacity;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyImageOpacity();
        }
      }
    },
    set imageOpacity( value ) { this.setImageOpacity( value ); },

    /**
     * Returns the opacity applied only to this image (not including children).
     * @public
     *
     * See setImageOpacity() documentation for more information.
     *
     * @returns {number}
     */
    getImageOpacity: function() {
      return this._imageOpacity;
    },
    get imageOpacity() { return this.getImageOpacity(); },

    /**
     * Provides an initial width for an image that has not loaded yet.
     * @public
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
     * @param {number} width - Expected width of the image's unloaded content
     * @returns {Image} - For chaining
     */
    setInitialWidth: function( width ) {
      assert && assert( typeof width === 'number' &&
                        width >= 0 &&
                        ( width % 1 === 0 ), 'initialWidth should be a non-negative integer' );
      if ( width !== this._initialWidth ) {
        this._initialWidth = width;

        this.invalidateImage();
      }

      return this;
    },
    set initialWidth( value ) { this.setInitialWidth( value ); },

    /**
     * Returns the initialWidth value set from setInitialWidth().
     * @public
     *
     * See setInitialWidth() for more documentation. A value of 0 is ignored.
     *
     * @returns {number}
     */
    getInitialWidth: function() {
      return this._initialWidth;
    },
    get initialWidth() { return this.getInitialWidth(); },

    /**
     * Provides an initial height for an image that has not loaded yet.
     * @public
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
     * @param {number} height - Expected height of the image's unloaded content
     * @returns {Image} - For chaining
     */
    setInitialHeight: function( height ) {
      assert && assert( typeof height === 'number' &&
                        height >= 0 &&
                        ( height % 1 === 0 ), 'initialHeight should be a non-negative integer' );
      if ( height !== this._initialHeight ) {
        this._initialHeight = height;

        this.invalidateImage();
      }

      return this;
    },
    set initialHeight( value ) { this.setInitialHeight( value ); },

    /**
     * Returns the initialHeight value set from setInitialHeight().
     * @public
     *
     * See setInitialHeight() for more documentation. A value of 0 is ignored.
     *
     * @returns {number}
     */
    getInitialHeight: function() {
      return this._initialHeight;
    },
    get initialHeight() { return this.getInitialHeight(); },

    /**
     * Sets whether mipmapping is supported.
     * @public
     *
     * This defaults to false, but is automatically set to true when a mipmap is provided to setImage(). Setting it to
     * true on non-mipmap images will trigger creation of a medium-quality mipmap that will be used.
     *
     * NOTE: This mipmap generation is slow and CPU-intensive. Providing precomputed mipmap resources to an Image node
     *       will be much faster, and of higher quality.
     *
     * @param {boolean} mipmap - Whether mipmapping is supported
     * @returns {Image} - For chaining
     */
    setMipmap: function( mipmap ) {
      assert && assert( typeof mipmap === 'boolean' );

      if ( this._mipmap !== mipmap ) {
        this._mipmap = mipmap;

        this.invalidateMipmaps();
      }

      return this;
    },
    set mipmap( value ) { this.setMipmap( value ); },

    /**
     * Returns whether mipmapping is supported.
     * @public
     *
     * See setMipmap() for more documentation.
     *
     * @returns {boolean}
     */
    isMipmap: function() {
      return this._mipmap;
    },
    get mipmap() { return this.isMipmap(); },

    /**
     * Sets how much level-of-detail is displayed for mipmapping.
     * @public
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
     *   mipmapLevel = Util.roundSymmetric( computedMipmapLevel + mipmapBias )
     *
     * @param bias
     * @returns {Image} - For chaining
     */
    setMipmapBias: function( bias ) {
      assert && assert( typeof bias === 'number' );

      if ( this._mipmapBias !== bias ) {
        this._mipmapBias = bias;

        this.invalidateMipmaps();
      }

      return this;
    },
    set mipmapBias( value ) { this.setMipmapBias( value ); },

    /**
     * Returns the current mipmap bias.
     * @public
     *
     * See setMipmapBias() for more documentation.
     *
     * @returns {number}
     */
    getMipmapBias: function() {
      return this._mipmapBias;
    },
    get mipmapBias() { return this.getMipmapBias(); },

    /**
     * The number of initial mipmap levels to compute (if Scenery generates the mipmaps by setting mipmap:true on a
     * non-mipmapped input).
     * @public
     *
     * @param {number} level - A non-negative integer representing the number of mipmap levels to precompute.
     * @returns {Image} - For chaining
     */
    setMipmapInitialLevel: function( level ) {
      assert && assert( typeof level === 'number' && level % 1 === 0 && level >= 0,
        'mipmapInitialLevel should be a non-negative integer' );

      if ( this._mipmapInitialLevel !== level ) {
        this._mipmapInitialLevel = level;

        this.invalidateMipmaps();
      }

      return this;
    },
    set mipmapInitialLevel( value ) { this.setMipmapInitialLevel( value ); },

    /**
     * Returns the current initial mipmap level.
     * @public
     *
     * See setMipmapInitialLevel() for more documentation.
     *
     * @returns {number}
     */
    getMipmapInitialLevel: function() {
      return this._mipmapInitialLevel;
    },
    get mipmapInitialLevel() { return this.getMipmapInitialLevel(); },

    /**
     * The maximum (lowest-resolution) level that Scenery will compute if it generates mipmaps (e.g. by setting
     * mipmap:true on a non-mipmapped input).
     * @public
     *
     * The default will precompute all default levels (from mipmapInitialLevel), so that we ideally don't hit mipmap
     * generation during animation.
     *
     * @param {number} level - A non-negative integer representing the maximum mipmap level to compute.
     * @returns {Image} - for Chaining
     */
    setMipmapMaxLevel: function( level ) {
      assert && assert( typeof level === 'number' && level % 1 === 0 && level >= 0,
        'mipmapMaxLevel should be a non-negative integer' );

      if ( this._mipmapMaxLevel !== level ) {
        this._mipmapMaxLevel = level;

        this.invalidateMipmaps();
      }

      return this;
    },
    set mipmapMaxLevel( value ) { this.setMipmapMaxLevel( value ); },

    /**
     * Returns the current maximum mipmap level.
     * @public
     *
     * See setMipmapMaxLevel() for more documentation.
     *
     * @returns {number}
     */
    getMipmapMaxLevel: function() {
      return this._mipmapMaxLevel;
    },
    get mipmapMaxLevel() { return this.getMipmapMaxLevel(); },

    /**
     * Constructs the next available (uncomputed) mipmap level, as long as the previous level was larger than 1x1.
     * @private
     */
    constructNextMipmap: function() {
      var level = this._mipmapCanvases.length;
      var biggerCanvas = this._mipmapCanvases[ level - 1 ];

      // ignore any 1x1 canvases (or smaller?!?)
      if ( biggerCanvas.width * biggerCanvas.height > 2 ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = Math.ceil( biggerCanvas.width / 2 );
        canvas.height = Math.ceil( biggerCanvas.height / 2 );

        // sanity check
        if ( canvas.width > 0 && canvas.height > 0 ) {
          // Draw half-scale into the smaller Canvas
          var context = canvas.getContext( '2d' );
          context.scale( 0.5, 0.5 );
          context.drawImage( biggerCanvas, 0, 0 );

          this._mipmapCanvases.push( canvas );
          this._mipmapURLs.push( canvas.toDataURL() );
        }
      }
    },

    /**
     * Triggers recomputation of mipmaps (as long as mipmapping is enabled)
     * @private
     */
    invalidateMipmaps: function() {
      // Clean output arrays
      cleanArray( this._mipmapCanvases );
      cleanArray( this._mipmapURLs );

      if ( this._image && this._mipmap ) {
        // If we have mipmap data as an input
        if ( this._mipmapData ) {
          for ( var k = 0; k < this._mipmapData.length; k++ ) {
            var url = this._mipmapData[ k ].url;
            this._mipmapURLs.push( url );
            this._mipmapData[ k ].updateCanvas && this._mipmapData[ k ].updateCanvas();
            this._mipmapCanvases.push( this._mipmapData[ k ].canvas );
          }
        }
        // Otherwise, we have an image (not mipmap) as our input, so we'll need to construct mipmap levels.
        else {
          var baseCanvas = document.createElement( 'canvas' );
          baseCanvas.width = this.getImageWidth();
          baseCanvas.height = this.getImageHeight();

          // if we are not loaded yet, just ignore
          if ( baseCanvas.width && baseCanvas.height ) {
            var baseContext = baseCanvas.getContext( '2d' );
            baseContext.drawImage( this._image, 0, 0 );

            this._mipmapCanvases.push( baseCanvas );
            this._mipmapURLs.push( baseCanvas.toDataURL() );

            var level = 0;
            while ( ++level < this._mipmapInitialLevel ) {
              this.constructNextMipmap();
            }
          }

          var stateLen = this._drawables.length;
          for ( var i = 0; i < stateLen; i++ ) {
            this._drawables[ i ].markDirtyMipmap();
          }
        }
      }

      this.trigger0( 'mipmap' );
    },

    /**
     * Returns the desired mipmap level (0-indexed) that should be used for the particular relative transform.
     * @public (scenery-internal)
     *
     * @param {Matrix3} matrix - The relative transformation matrix of the node.
     */
    getMipmapLevel: function( matrix ) {
      assert && assert( this._mipmap, 'Assumes mipmaps can be used' );

      // a sense of "average" scale, which should be exact if there is no asymmetric scale/shear applied
      var scale = ( Math.sqrt( matrix.m00() * matrix.m00() + matrix.m10() * matrix.m10() ) +
                    Math.sqrt( matrix.m01() * matrix.m01() + matrix.m11() * matrix.m11() ) ) / 2;
      scale *= ( window.devicePixelRatio || 1 ); // for retina-like devices

      assert && assert( typeof scale === 'number' && scale > 0, 'scale should be a positive number' );

      // If we are shown larger than scale, ALWAYS choose the highest resolution
      if ( scale >= 1 ) {
        return 0;
      }

      var level = log2( 1 / scale ); // our approximate level of detail
      level = Util.roundSymmetric( level + this._mipmapBias - 0.7 ); // convert to an integer level (-0.7 is a good default)

      if ( level < 0 ) {
        level = 0;
      }
      if ( level > this._mipmapMaxLevel ) {
        level = this._mipmapMaxLevel;
      }

      // If necessary, do lazy construction of the mipmap level
      if ( this.mipmap && !this._mipmapCanvases[ level ] ) {
        var currentLevel = this._mipmapCanvases.length - 1;
        while ( ++currentLevel <= level ) {
          this.constructNextMipmap();
        }
        // Sanity check, since constructNextMipmap() may have had to bail out. We had to compute some, so use the last
        return Math.min( level, this._mipmapCanvases.length - 1 );
      }
      // Should already be constructed, or isn't needed
      else {
        return level;
      }
    },

    /**
     * Returns a matching Canvas element for the given level-of-detail.
     * @public (scenery-internal)
     *
     * @param {number} level - Non-negative integer representing the mipmap level
     * @returns {HTMLCanvasElement} - Matching <canvas> for the level of detail
     */
    getMipmapCanvas: function( level ) {
      assert && assert( typeof level === 'number' &&
                        level >= 0 &&
                        level < this._mipmapCanvases.length &&
                        ( level % 1 ) === 0 );

      // Sanity check to make sure we have copied the image data in if necessary.
      if ( this._mipmapData ) {
        // level may not exist (it was generated), and updateCanvas may not exist
        this._mipmapData[ level ] && this._mipmapData[ level ].updateCanvas && this._mipmapData[ level ].updateCanvas();
      }
      return this._mipmapCanvases[ level ];
    },

    /**
     * Returns a matching URL string for an image for the given level-of-detail.
     * @public (scenery-internal)
     *
     * @param {number} level - Non-negative integer representing the mipmap level
     * @returns {string} - Matching data URL for the level of detail
     */
    getMipmapURL: function( level ) {
      assert && assert( typeof level === 'number' &&
                        level >= 0 &&
                        level < this._mipmapCanvases.length &&
                        ( level % 1 ) === 0 );

      return this._mipmapURLs[ level ];
    },

    /**
     * Returns whether there are mipmap levels that have been computed.
     * @public (scenery-internal)
     *
     * @returns {boolean}
     */
    hasMipmaps: function() {
      return this._mipmapCanvases.length > 0;
    },

    /**
     * Returns the width of the displayed image (not related to how this node is transformed).
     * @public
     *
     * NOTE: If the image is not loaded and an initialWidth was provided, that width will be used.
     *
     * @returns {number}
     */
    getImageWidth: function() {
      var detectedWidth = this._mipmapData ? this._mipmapData[ 0 ].width : ( this._image.naturalWidth || this._image.width );
      if ( detectedWidth === 0 ) {
        return this._initialWidth; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialWidth === 0 || this._initialWidth === detectedWidth, 'Bad Image.initialWidth' );

        return detectedWidth;
      }
    },
    get imageWidth() { return this.getImageWidth(); },

    /**
     * Returns the height of the displayed image (not related to how this node is transformed).
     * @public
     *
     * NOTE: If the image is not loaded and an initialHeight was provided, that height will be used.
     *
     * @returns {number}
     */
    getImageHeight: function() {
      var detectedHeight = this._mipmapData ? this._mipmapData[ 0 ].height : ( this._image.naturalHeight || this._image.height );
      if ( detectedHeight === 0 ) {
        return this._initialHeight; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialHeight === 0 || this._initialHeight === detectedHeight, 'Bad Image.initialHeight' );

        return detectedHeight;
      }
    },
    get imageHeight() { return this.getImageHeight(); },

    /**
     * If our provided image is an HTMLImageElement, returns its URL (src).
     * @public (scenery-internal)
     *
     * @returns {string}
     */
    getImageURL: function() {
      assert && assert( this._image instanceof HTMLImageElement, 'Only supported for HTML image elements' );

      return this._image.src;
    },

    /**
     * Whether this Node itself is painted (displays something itself).
     * @public
     * @override
     *
     * @returns {boolean}
     */
    isPainted: function() {
      // Always true for Image nodes
      return true;
    },

    /**
     * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
     * coordinate frame of this node.
     * @protected
     * @override
     *
     * @param {CanvasContextWrapper} wrapper
     * @param {Matrix3} matrix - The transformation matrix already applied to the context.
     */
    canvasPaintSelf: function( wrapper, matrix ) {
      //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
      ImageCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
    },

    /**
     * Creates a DOM drawable for this Image.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {DOMSelfDrawable}
     */
    createDOMDrawable: function( renderer, instance ) {
      return ImageDOMDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a SVG drawable for this Image.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return ImageSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Image.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return ImageCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a WebGL drawable for this Image.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return ImageWebGLDrawable.createFromPool( renderer, instance );
    },

    /**
     * Attaches our on-load listener to our current image.
     * @private
     */
    attachImageLoadListener: function() {
      assert && assert( !this._imageLoadListenerAttached, 'Should only be attached to one thing at a time' );

      if ( !this.isDisposed ) {
        this._image.addEventListener( 'load', this._imageLoadListener );
        this._imageLoadListenerAttached = true;
      }
    },

    /**
     * Detaches our on-load listener from our current image.
     * @private
     */
    detachImageLoadListener: function() {
      assert && assert( this._imageLoadListenerAttached, 'Needs to be attached first to be detached.' );

      this._image.removeEventListener( 'load', this._imageLoadListener );
      this._imageLoadListenerAttached = false;
    },

    /**
     * Called when our image has loaded (it was not yet loaded with then listener was added)
     * @private
     */
    onImageLoad: function() {
      assert && assert( this._imageLoadListenerAttached, 'If onImageLoad is firing, it should be attached' );

      this.invalidateImage();
      this.detachImageLoadListener();
    },

    /**
     * Disposes the path, releasing image listeners if needed (and preventing new listeners from being added).
     * @public
     * @override
     */
    dispose: function() {
      if ( this._image && this._imageLoadListenerAttached ) {
        this.detachImageLoadListener();
      }

      Node.prototype.dispose.call( this );
    }
  } );

  /**
   * Creates an SVG image element with a given URL and dimensions
   * @public
   *
   * @param {string} url - The URL for the image
   * @param {number} width - Non-negative integer for the image's width
   * @param {number} height - Non-negative integer for the image's height
   * @returns {SVGImageElement}
   */
  Image.createSVGImage = function( url, width, height ) {
    assert && assert( typeof url === 'string', 'Requires the URL as a string' );
    assert && assert( typeof width === 'number' && isFinite( width ) && width >= 0 && ( width % 1 ) === 0,
      'width should be a non-negative finite integer' );
    assert && assert( typeof height === 'number' && isFinite( height ) && height >= 0 && ( height % 1 ) === 0,
      'height should be a non-negative finite integer' );

    var element = document.createElementNS( scenery.svgns, 'image' );
    element.setAttribute( 'x', 0 );
    element.setAttribute( 'y', 0 );
    element.setAttribute( 'width', width + 'px' );
    element.setAttribute( 'height', height + 'px' );
    element.setAttributeNS( scenery.xlinkns, 'xlink:href', url );

    return element;
  };

  /**
   * Creates an object suitable to be passed to Image as a mipmap (from a Canvas)
   * @public
   *
   * @param {HTMLCanvasElement} baseCanvas
   * @returns {Array}
   */
  Image.createFastMipmapFromCanvas = function( baseCanvas ) {
    var mipmaps = [];

    var baseURL = baseCanvas.toDataURL();
    var baseImage = new window.Image();
    baseImage.src = baseURL;

    // base level
    mipmaps.push( {
      img: baseImage,
      url: baseURL,
      width: baseCanvas.width,
      height: baseCanvas.height,
      canvas: baseCanvas
    } );

    var largeCanvas = baseCanvas;
    while ( largeCanvas.width >= 2 && largeCanvas.height >= 2 ) {
      // smaller level
      var mipmap = {};

      // draw half-size
      var canvas = document.createElement( 'canvas' );
      canvas.width = mipmap.width = Math.ceil( largeCanvas.width / 2 );
      canvas.height = mipmap.height = Math.ceil( largeCanvas.height / 2 );
      var context = canvas.getContext( '2d' );
      context.setTransform( 0.5, 0, 0, 0.5, 0, 0 );
      context.drawImage( largeCanvas, 0, 0 );

      // set up the image and url
      mipmap.canvas = canvas;
      mipmap.url = canvas.toDataURL();
      mipmap.img = new window.Image();
      mipmap.img.src = mipmap.url;
      largeCanvas = canvas;

      mipmaps.push( mipmap );
    }

    return mipmaps;
  };

  // @public {Object} - Initial values for most Node mutator options
  Image.DEFAULT_OPTIONS = _.extend( {}, Node.DEFAULT_OPTIONS, DEFAULT_OPTIONS );

  return Image;
} );
