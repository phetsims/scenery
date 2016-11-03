// Copyright 2013-2015, University of Colorado Boulder

/**
 * A node that displays a single image either from an actual HTMLImageElement, a URL, a Canvas element, or a mipmap
 * data structure described in the constructor.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var platform = require( 'PHET_CORE/platform' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  var Renderer = require( 'SCENERY/display/Renderer' ); // we need to specify the Renderer in the prototype
  require( 'SCENERY/util/Util' );

  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var SpriteSheet = require( 'SCENERY/util/SpriteSheet' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMImageElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGImageElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  var defaultMipmapBias = -0.7;
  var defaultMipmapInitialLevel = 4; // by default, precompute all levels that will be used (so we don't hit this during animation)
  var defaultMipmapMaxLevel = 5;

  var log2 = Math.log2 || function( x ) {
      return Math.log( x ) / Math.LN2;
    };

  /*
   * Constructs an Image node from a particular source.
   * @public
   *
   * We support a few different 'image' parameter types:
   *
   * HTMLImageElement - A normal HTML <img>. If it hasn't been fully loaded yet, Scenery will take care of adding a
   *   listener that will update Scenery with its width/height (and load its data) when the image is fully loaded. NOTE
   *   that if you just created the <img>, it probably isn't loaded yet, particularly in Safari. If the Image node is
   *   constructed with an <img> that hasn't fully loaded, it will have a width and height of 0, which may cause issues
   *   if you are using bounds for layout. Please see initialWidth/initialHeight notes below.
   *
   * URL - Provide a {string}, and Scenery will assume it is a URL. This can be a normal URL, or a data URI, both will
   *   work. Please note that this has the same loading-order issues as using HTMLImageElement, but that it's almost
   *   always guaranteed to not have a width/height when you create the Image node. Note that data URI support for
   *   formats depends on the browser - only JPEG and PNG are supported broadly. Please see initialWidth/initialHeight
   *   notes below.
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
   *     height: {number} // height of the mipmap level, in pixels
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
   *
   * -----------------
   *
   * Image also supports the following options beyond what Node itself provides:
   *
   * image - {see above} Allows changing the underlying input image to a Scenery Image node. Note that if for some
   *    reason the provided image was mutated somehow, it's recommended to call invalidateImage() instead of changing
   *    the image reference.
   *
   * initialWidth - {number} If the input image hasn't loaded yet, but the (expected) size is known, providing an
   *    initialWidth will cause the Image node to have the correct bounds (width) before the pixel data has been fully
   *    loaded. A value of 0 will be ignored.
   *    NOTE: setImage will reset this value to 0 (ingnored), since it's potentially likely the new image has
   *    different dimensions than the current image.
   *
   * initialHeight - {number} If the input image hasn't loaded yet, but the (expected) size is known, providing an
   *    initialHeight will cause the Image node to have the correct bounds (height) before the pixel data has been fully
   *    loaded. A value of 0 will be ignored.
   *    NOTE: setImage will reset this value to 0 (ingnored), since it's potentially likely the new image has
   *    different dimensions than the current image.
   *
   * mipmap - {boolean} Whether mipmaps are supported. Defaults to false, but is automatically set to true when a mipmap
   *    image is provided to it. Setting it to true on non-mipmap images will trigger creation of a medium-quality
   *    mipmap that will be used. NOTE that this mipmap generation is slow and CPU-intensive. Providing precomputed
   *    mipmap resources to an Image node will be much faster, and of higher quality.
   *
   * mipmapBias - {number} Allows adjustment of how much level-of-detail is displayed. Increasing it will typically
   *    decrease the displayed resolution, and decreases (going negative) will increase the displayed resolution, such
   *    that approximately:
   *        mipmapLevel = Math.round( computedMipmapLevel + mipmapBias )
   *
   * mipmapInitialLevel - {number} If relying on Scenery to generate the mipmaps (mipmap:true on a non-mipmap input),
   *    this will be the number of initial levels to compute.
   *
   * mipmapMaxLevel - {number} If relying on Scenery to generate the mipmaps (mipmap:true on a non-mipmap input),
   *    this will be the maximum (smallest) level that Scenery will compute.
   *
   * @param {see above} image
   * @param {Object} [options]
   */
  function Image( image, options ) {
    assert && assert( image, 'image should be available' );

    // allow not passing an options object
    options = options || {};

    // rely on the setImage call from the super constructor to do the setup
    if ( image ) {
      options.image = image;
    }

    // When non-zero, overrides the Image's natural width/height (in the local coordinate frame) while the Image's
    // dimensions can't be detected yet (i.e. it reports 0x0 like Safari does for an image that isn't fully loaded).
    // This allows for faster display of dynamically-created images if the dimensions are known ahead-of-time.
    // If the intitial dimensions don't match the image's dimensions after it is loaded, an assertion will be fired.
    this._initialWidth = 0;
    this._initialHeight = 0;

    // {number} - Opacity applied directly to the image itself, without affecting the rendering of children.
    this._imageOpacity = 1;

    // Mipmap client values
    this._mipmap = false; // {bool} - Whether mipmapping is enabled
    this._mipmapBias = defaultMipmapBias; // {number} - Amount of level-of-detail adjustment added to everything.
    this._mipmapInitialLevel = defaultMipmapInitialLevel; // {number} - Quantity of mipmap levels to initially compute
    this._mipmapMaxLevel = defaultMipmapMaxLevel; // {number} - Maximum mipmap levels to compute (lazily if > initial)

    // Mipmap internal handling
    this._mipmapCanvases = []; // TODO: power-of-2 handling for WebGL if helpful
    this._mipmapURLs = [];
    this._mipmapData = null; // if mipmap data is passed into our Image, it will be stored here for processing

    var self = this;
    // allows us to invalidate our bounds whenever an image is loaded
    this.loadListener = function( event ) {
      self.invalidateImage();

      // don't leak memory!
      self._image.removeEventListener( 'load', self.loadListener );
    };

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
    _mutatorKeys: [ 'image', 'imageOpacity', 'initialWidth', 'initialHeight',
                    'mipmap', 'mipmapBias', 'mipmapInitialLevel', 'mipmapMaxLevel' ].concat( Node.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Node.prototype.drawableMarkFlags.concat( [ 'image', 'imageOpacity', 'mipmap' ] ),

    allowsMultipleDOMInstances: false, // TODO: support multiple instances

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

    getImage: function() {
      return this._image;
    },
    get image() { return this.getImage(); },

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
     * Sets the current image to be displayed by this Image node.
     *
     * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image - See the constructor documentation for more
     *                                                                  information about supported types of input
     *                                                                  images, and their possible performance
     *                                                                  implications.
     * @returns {Image} - Self reference for chaining
     */
    setImage: function( image ) {
      assert && assert( image, 'image should be available' );
      assert && assert( typeof image === 'string' ||
                        image instanceof HTMLImageElement ||
                        image instanceof HTMLCanvasElement ||
                        image instanceof Array, 'image is not of the correct type' );

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

        // don't leak memory by referencing old images
        if ( this._image ) {
          this._image.removeEventListener( 'load', this.loadListener );
        }

        // clear old mipmap data references
        this._mipmapData = null;

        if ( typeof image === 'string' ) {
          // create an image with the assumed URL
          var src = image;
          image = document.createElement( 'img' );
          image.addEventListener( 'load', this.loadListener );
          image.src = src;
        }
        else if ( image instanceof HTMLImageElement ) {
          // only add a listener if we probably haven't loaded yet
          if ( !image.width || !image.height ) {
            image.addEventListener( 'load', this.loadListener );
          }
        }
        else if ( image instanceof Array ) {
          // mipmap data!
          this._mipmapData = image;
          image = image[ 0 ].img; // presumes we are already loaded

          // force initialization of mipmapping parameters, since invalidateMipmaps() is guaranteed to run below
          this._mipmapInitialLevel = this._mipmapMaxLevel = this._mipmapData.length;
          this._mipmap = true;
        }

        this._image = image;

        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },
    set image( value ) { this.setImage( value ); },

    /**
     * Sets the image with specific dimensions.
     * @param {*} image - see the constructor
     * @param {number} width - width of image
     * @param {number} height - height of image
     * @public
     */
    setImageWithSize: function( image, width, height ) {
      // First, setImage(), as it will reset the initial width and height
      this.setImage( image );

      // Then apply the initial dimensions
      this.setInitialWidth( width );
      this.setInitialHeight( height );
    },

    /**
     * Returns the opacity applied only to this image (not including children).
     * @public
     *
     * @returns {number}
     */
    getImageOpacity: function() {
      return this._imageOpacity;
    },
    get imageOpacity() { return this.getImageOpacity(); },

    /**
     * Sets an opacity that is applied only to this image (will not affect children or the rest of the node's subtree).
     * @public
     *
     * @param {number} imageOpacty - Should be a number between 0 (transparent) and 1 (opaque), just like normal opacity
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

    getInitialWidth: function() {
      return this._initialWidth;
    },
    get initialWidth() { return this.getInitialWidth(); },

    setInitialWidth: function( width ) {
      assert && assert( typeof width === 'number' &&
                        width >= 0 &&
                        ( width % 1 === 0 ), 'initialWidth should be a non-negative integer' );
      if ( width !== this._initialWidth ) {
        this._initialWidth = width;

        this.invalidateImage();
      }
    },
    set initialWidth( value ) { this.setInitialWidth( value ); },

    getInitialHeight: function() {
      return this._initialHeight;
    },
    get initialHeight() { return this.getInitialHeight(); },

    setInitialHeight: function( height ) {
      assert && assert( typeof height === 'number' &&
                        height >= 0 &&
                        ( height % 1 === 0 ), 'initialHeight should be a non-negative integer' );
      if ( height !== this._initialHeight ) {
        this._initialHeight = height;

        this.invalidateImage();
      }
    },
    set initialHeight( value ) { this.setInitialHeight( value ); },

    isMipmap: function() {
      return this._mipmap;
    },
    get mipmap() { return this.isMipmap(); },

    setMipmap: function( mipmap ) {
      assert && assert( typeof mipmap === 'boolean' );

      if ( this._mipmap !== mipmap ) {
        this._mipmap = mipmap;

        this.invalidateMipmaps();
      }
    },
    set mipmap( value ) { this.setMipmap( value ); },

    getMipmapBias: function() {
      return this._mipmapBias;
    },
    get mipmapBias() { return this.getMipmapBias(); },

    setMipmapBias: function( bias ) {
      assert && assert( typeof bias === 'number' );

      if ( this._mipmapBias !== bias ) {
        this._mipmapBias = bias;

        this.invalidateMipmaps();
      }
    },
    set mipmapBias( value ) { this.setMipmapBias( value ); },

    getMipmapInitialLevel: function() {
      return this._mipmapInitialLevel;
    },
    get mipmapInitialLevel() { return this.getMipmapInitialLevel(); },

    setMipmapInitialLevel: function( level ) {
      assert && assert( typeof level === 'number' );

      if ( this._mipmapInitialLevel !== level ) {
        this._mipmapInitialLevel = level;

        this.invalidateMipmaps();
      }
    },
    set mipmapInitialLevel( value ) { this.setMipmapInitialLevel( value ); },

    getMipmapMaxLevel: function() {
      return this._mipmapMaxLevel;
    },
    get mipmapMaxLevel() { return this.getMipmapMaxLevel(); },

    setMipmapMaxLevel: function( level ) {
      assert && assert( typeof level === 'number' );

      if ( this._mipmapMaxLevel !== level ) {
        this._mipmapMaxLevel = level;

        this.invalidateMipmaps();
      }
    },
    set mipmapMaxLevel( value ) { this.setMipmapMaxLevel( value ); },

    // @private
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
          var context = canvas.getContext( '2d' );
          context.scale( 0.5, 0.5 );
          context.drawImage( biggerCanvas, 0, 0 );

          this._mipmapCanvases.push( canvas );
          this._mipmapURLs.push( canvas.toDataURL() );
        }
      }
    },

    // @public
    invalidateMipmaps: function() {
      cleanArray( this._mipmapCanvases );
      cleanArray( this._mipmapURLs );

      if ( this._image && this._mipmap ) {
        if ( this._mipmapData ) {
          for ( var k = 0; k < this._mipmapData.length; k++ ) {
            var url = this._mipmapData[ k ].url;
            this._mipmapURLs.push( url );
            // TODO: baseCanvas only upon demand?
            var canvas = document.createElement( 'canvas' );
            canvas.width = this._mipmapData[ k ].width;
            canvas.height = this._mipmapData[ k ].height;
            var context = canvas.getContext( '2d' );
            context.drawImage( this._mipmapData[ k ].img, 0, 0 );
            this._mipmapCanvases.push( canvas );
          }
        }
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
     * Returns the desired mipmap level (0-indexed) that should be used for the particular scale.
     *
     * @param {number} scale
     */
    getMipmapLevel: function( scale ) {
      assert && assert( scale > 0 );

      // If we are shown larger than scale, ALWAYS choose the highest resolution
      if ( scale >= 1 ) {
        return 0;
      }

      var level = log2( 1 / scale ); // our approximate level of detail
      level = Math.round( level + this._mipmapBias ); // convert to an integer level

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
     * @returns {HTMLCanvasElement} - Matching <canvas> for the level of detail
     */
    getMipmapCanvas: function( level ) {
      assert && assert( level >= 0 && level < this._mipmapCanvases.length && ( level % 1 ) === 0 );

      return this._mipmapCanvases[ level ];
    },

    /**
     * @returns {string} - Matching data URL for the level of detail
     */
    getMipmapURL: function( level ) {
      assert && assert( level >= 0 && level < this._mipmapCanvases.length && ( level % 1 ) === 0 );

      return this._mipmapURLs[ level ];
    },

    hasMipmaps: function() {
      return this._mipmapCanvases.length > 0;
    },

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

    getImageURL: function() {
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
     */
    canvasPaintSelf: function( wrapper ) {
      Image.ImageCanvasDrawable.prototype.paintCanvas( wrapper, this );
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
      return Image.ImageDOMDrawable.createFromPool( renderer, instance );
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
      return Image.ImageSVGDrawable.createFromPool( renderer, instance );
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
      return Image.ImageCanvasDrawable.createFromPool( renderer, instance );
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
      return Image.ImageWebGLDrawable.createFromPool( renderer, instance );
    },

    /**
     * Returns a string containing constructor information for Node.string().
     * @protected
     * @override
     *
     * @param {string} propLines - A string representing the options properties that need to be set.
     * @returns {string}
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + ( this._image.src ? this._image.src.replace( /'/g, '\\\'' ) : 'other' ) + '\', {' + propLines + '} )';
    }
  } );

  // utility for others
  Image.createSVGImage = function( url, width, height ) {
    var element = document.createElementNS( scenery.svgns, 'image' );
    element.setAttribute( 'x', 0 );
    element.setAttribute( 'y', 0 );
    element.setAttribute( 'width', width + 'px' );
    element.setAttribute( 'height', height + 'px' );
    element.setAttributeNS( scenery.xlinkns, 'xlink:href', url );

    return element;
  };

  // Creates an {Object} suitable to be passed to Image as a mipmap (from a Canvas)
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
      height: baseCanvas.height
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
      mipmap.url = canvas.toDataURL();
      mipmap.img = new window.Image();
      mipmap.img.src = mipmap.url;
      largeCanvas = canvas;

      mipmaps.push( mipmap );
    }

    return mipmaps;
  };

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG) //TODO: Does this also apply to WebGL?
   *----------------------------------------------------------------------------*/

  /**
   * A mixin to drawables for Image that need to store state about what the current display is currently showing,
   * so that updates to the Image will only be made on attributes that specifically changed (and no change will be
   * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
   * for DOM and SVG drawables.
   */
  Image.ImageStatefulDrawable = {
    /**
     * Given the type (constructor) of a drawable, we'll mix in a combination of:
     * - initialization/disposal with the *State suffix
     * - mark* methods to be called on all drawables of nodes of this type, that set specific dirty flags
     *
     * This will allow drawables that mix in this type to do the following during an update:
     * 1. Check specific dirty flags (e.g. if the fill changed, update the fill of our SVG element).
     * 2. Call setToCleanState() once done, to clear the dirty flags.
     *
     * @param {function} drawableType - The constructor for the drawable type
     */
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      /**
       * Initializes the stateful mixin state, starting its "lifetime" until it is disposed with disposeState().
       * @protected
       *
       * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
       * @param {Instance} instance
       * @returns {ImageStatefulDrawable} - Self reference for chaining
       */
      proto.initializeState = function( renderer, instance ) {
        // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
        //                        need to accelerate the transform case.
        this.paintDirty = true;
        this.dirtyImage = true;
        this.dirtyImageOpacity = true;
        this.dirtyMipmap = true;

        return this; // allow for chaining
      };

      /**
       * Disposes the stateful mixin state, so it can be put into the pool to be initialized again.
       * @protected
       */
      proto.disposeState = function() {

      };

      /**
       * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
       * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
       * @public (scenery-internal)
       *
       * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
       * dirty.
       */
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyImage = function() {
        this.dirtyImage = true;
        this.markPaintDirty();
      };

      proto.markDirtyImageOpacity = function() {
        this.dirtyImageOpacity = true;
        this.markPaintDirty();
      };

      proto.markDirtyMipmap = function() {
        this.dirtyMipmap = true;
        this.markPaintDirty();
      };

      /**
       * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
       * @public (scenery-internal)
       */
      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyImage = false;
        this.dirtyImageOpacity = false;
        this.dirtyMipmap = false;
      };
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of a Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Image.ImageDOMDrawable = function ImageDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( DOMSelfDrawable, Image.ImageDOMDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {ImageDOMDrawable} - Self reference for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeDOMSelfDrawable( renderer, instance );

      // Stateful mix-in initialization
      this.initializeState( renderer, instance );

      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.domElement ) {
        this.domElement = document.createElement( 'img' );
        this.domElement.style.display = 'block';
        this.domElement.style.position = 'absolute';
        this.domElement.style.pointerEvents = 'none';
        this.domElement.style.left = '0';
        this.domElement.style.top = '0';
      }

      // Whether we have an opacity attribute specified on the DOM element.
      this.hasOpacity = false;

      // Apply CSS needed for future CSS transforms to work properly.
      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    /**
     * Updates our DOM element so that its appearance matches our node's representation.
     * @protected
     *
     * This implements part of the DOMSelfDrawable required API for subtypes.
     */
    updateDOM: function() {
      var node = this.node;
      var img = this.domElement;

      if ( this.paintDirty && this.dirtyImage ) {
        // TODO: allow other ways of showing a DOM image?
        img.src = node._image ? node._image.src : '//:0'; // NOTE: for img with no src (but with a string), see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
      }

      if ( this.dirtyImageOpacity ) {
        if ( node._imageOpacity === 1 ) {
          if ( this.hasOpacity ) {
            this.hasOpacity = false;
            img.style.opacity = '';
          }
        }
        else {
          this.hasOpacity = true;
          img.style.opacity = node._imageOpacity;
        }
      }

      if ( this.transformDirty ) {
        scenery.Util.applyPreparedTransform( this.getTransformMatrix(), this.domElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToCleanState();
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.disposeState();

      if ( !keepDOMImageElements ) {
        this.domElement = null; // clear our DOM reference if we want to toss it
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageDOMDrawable );
  // This sets up ImageDOMDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Image.ImageDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of a Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Image.ImageSVGDrawable = function ImageSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Image.ImageSVGDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {ImageSVGDrawable} - Self reference for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeSVGSelfDrawable( renderer, instance, false, keepSVGImageElements ); // usesPaint: false

      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' initialized for ' + instance.toString() );
      var self = this;

      // @protected {SVGImageElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
      this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'image' );
      this.svgElement.setAttribute( 'x', 0 );
      this.svgElement.setAttribute( 'y', 0 );

      // Whether we have an opacity attribute specified on the DOM element.
      this.hasOpacity = false;

      this._usingMipmap = false;
      this._mipmapLevel = -1; // will always be invalidated

      // if mipmaps are enabled, this listener will be added to when our relative transform changes
      this._mipmapTransformListener = this._mipmapTransformListener || function() {
          sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( self.id + ' Transform dirties mipmap' );
          self.markDirtyMipmap();
        };

      this._mipmapListener = this._mipmapListener || function() {
          // sanity check
          self.markDirtyMipmap();

          // update our mipmap usage status
          self.updateMipmapStatus( self.node._mipmap );
        };
      this.node.onStatic( 'mipmap', this._mipmapListener );
      this.updateMipmapStatus( instance.node._mipmap );

      return this;
    },

    /**
     * Updates the SVG elements so that they will appear like the current node's representation.
     * @protected
     *
     * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
     */
    updateSVGSelf: function() {
      var image = this.svgElement;

      if ( this.dirtyImage ) {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Updating dirty image' );
        if ( this.node._image ) {
          // like <image xlink:href='http://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
          this.updateURL( image, true );
        }
        else {
          image.setAttribute( 'width', '0' );
          image.setAttribute( 'height', '0' );
          image.setAttributeNS( scenery.xlinkns, 'xlink:href', '//:0' ); // see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
        }
      }
      else if ( this.dirtyMipmap && this.node._image ) {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Updating dirty mipmap' );
        this.updateURL( image, false );
      }

      if ( this.dirtyImageOpacity ) {
        if ( this.node._imageOpacity === 1 ) {
          if ( this.hasOpacity ) {
            this.hasOpacity = false;
            image.removeAttribute( 'opacity' );
          }
        }
        else {
          this.hasOpacity = true;
          image.setAttribute( 'opacity', this.node._imageOpacity );
        }
      }
    },

    updateURL: function( image, forced ) {
      // determine our mipmap level, if any is used
      var level = -1; // signals a default of "we are not using mipmapping"
      if ( this.node._mipmap ) {
        var matrix = this.instance.relativeTransform.matrix;
        // a sense of "average" scale, which should be exact if there is no asymmetric scale/shear applied
        var approximateScale = ( Math.sqrt( matrix.m00() * matrix.m00() + matrix.m10() * matrix.m10() ) +
                                 Math.sqrt( matrix.m01() * matrix.m01() + matrix.m11() * matrix.m11() ) ) / 2;
        approximateScale *= ( window.devicePixelRatio || 1 ); // for retina-like devices
        level = this.node.getMipmapLevel( approximateScale );
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Mipmap level: ' + level );
      }

      // bail out if we would use the currently-used mipmap level (or none) and there was no image change
      if ( !forced && level === this._mipmapLevel ) {
        return;
      }

      // if we are switching to having no mipmap
      if ( this._mipmapLevel >= 0 && level === -1 ) {
        // IE guard needed since removeAttribute fails, see https://github.com/phetsims/scenery/issues/395
        ( platform.ie9 || platform.ie10 ) ? image.setAttribute( 'transform', '' ) : image.removeAttribute( 'transform' );
      }
      this._mipmapLevel = level;

      if ( this.node._mipmap && this.node.hasMipmaps() ) {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Setting image URL to mipmap level ' + level );
        var url = this.node.getMipmapURL( level );
        var canvas = this.node.getMipmapCanvas( level );
        image.setAttribute( 'width', canvas.width + 'px' );
        image.setAttribute( 'height', canvas.height + 'px' );
        // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
        // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
        // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
        // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
        image.setAttribute( 'transform', 'scale(' + Math.pow( 2, level ).toFixed( 20 ) + ')' );
        image.setAttributeNS( scenery.xlinkns, 'xlink:href', url );
      }
      else {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Setting image URL' );
        image.setAttribute( 'width', this.node.getImageWidth() + 'px' );
        image.setAttribute( 'height', this.node.getImageHeight() + 'px' );
        image.setAttributeNS( scenery.xlinkns, 'xlink:href', this.node.getImageURL() );
      }
    },

    updateMipmapStatus: function( usingMipmap ) {
      if ( this._usingMipmap !== usingMipmap ) {
        this._usingMipmap = usingMipmap;

        if ( usingMipmap ) {
          sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Adding mipmap compute/listener needs' );
          this.instance.relativeTransform.addListener( this._mipmapTransformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
          this.instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
        }
        else {
          sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' Removing mipmap compute/listener needs' );
          this.instance.relativeTransform.removeListener( this._mipmapTransformListener );
          this.instance.relativeTransform.removePrecompute();
        }

        // sanity check
        this.markDirtyMipmap();
      }
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' disposing' );

      // clean up mipmap listeners and compute needs
      this.updateMipmapStatus( false );
      this.node.offStatic( 'mipmap', this._mipmapListener );

      SVGSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageSVGDrawable );
  // This sets up ImageSVGDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Image.ImageSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of a Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Image.ImageCanvasDrawable = function ImageCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Image.ImageCanvasDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {ImageCanvasDrawable} - For chaining
     */
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    /**
     * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
     * @public
     *
     * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
     * other required effects (opacity, clipping, etc.) have already been prepared.
     *
     * This is part of the CanvasSelfDrawable API required to be implemented for subtypes.
     *
     * @param {CanvasContextWrapper} wrapper - Contains the Canvas and its drawing context
     * @param {Node} node - Our node that is being drawn
     */
    paintCanvas: function( wrapper, node ) {
      var hasImageOpacity = node._imageOpacity !== 1;

      // Ensure that the image has been loaded by checking whether it has a width or height of 0.
      // See https://github.com/phetsims/scenery/issues/536
      if ( node._image && node._image.width !== 0 && node._image.height !== 0 ) {
        // If we have image opacity, we need to apply the opacity on top of whatever globalAlpha may exist
        if ( hasImageOpacity ) {
          wrapper.context.save();
          wrapper.context.globalAlpha *= node._imageOpacity;
        }

        wrapper.context.drawImage( node._image, 0, 0 );

        if ( hasImageOpacity ) {
          wrapper.context.restore();
        }
      }
    },

    // stateless dirty functions
    markDirtyImage: function() { this.markPaintDirty(); },
    markDirtyMipmap: function() { this.markPaintDirty(); },
    markDirtyImageOpacity: function() { this.markPaintDirty(); }
  } );
  // This sets up ImageCanvasDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Image.ImageCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  // For alignment, we keep things to 8 components, aligned on 4-byte boundaries.
  // See https://developer.apple.com/library/ios/documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/TechniquesforWorkingwithVertexData/TechniquesforWorkingwithVertexData.html#//apple_ref/doc/uid/TP40008793-CH107-SW15
  var WEBGL_COMPONENTS = 5; // format [X Y U V A] for 6 vertices

  var VERTEX_0_OFFSET = WEBGL_COMPONENTS * 0;
  var VERTEX_1_OFFSET = WEBGL_COMPONENTS * 1;
  var VERTEX_2_OFFSET = WEBGL_COMPONENTS * 2;
  var VERTEX_3_OFFSET = WEBGL_COMPONENTS * 3;
  var VERTEX_4_OFFSET = WEBGL_COMPONENTS * 4;
  var VERTEX_5_OFFSET = WEBGL_COMPONENTS * 5;

  var VERTEX_X_OFFSET = 0;
  var VERTEX_Y_OFFSET = 1;
  var VERTEX_U_OFFSET = 2;
  var VERTEX_V_OFFSET = 3;
  var VERTEX_A_OFFSET = 4;

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of an Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Image.ImageWebGLDrawable = function ImageWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( WebGLSelfDrawable, Image.ImageWebGLDrawable, {
    // TODO: doc
    webglRenderer: Renderer.webglTexturedTriangles,

    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {ImageWebGLDrawable} - Self reference for chaining
     */
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );

      if ( !this.vertexArray ) {
        // for 6 vertices
        this.vertexArray = new Float32Array( WEBGL_COMPONENTS * 6 ); // 5-length components for 6 vertices (2 tris).
      }

      // corner vertices in the relative transform root coordinate space
      this.upperLeft = new Vector2();
      this.lowerLeft = new Vector2();
      this.upperRight = new Vector2();
      this.lowerRight = new Vector2();

      this.xyDirty = true; // is our vertex position information out of date?
      this.uvDirty = true; // is our UV information out of date?
      this.updatedOnce = false;

      // {SpriteSheet.Sprite} exported for WebGLBlock's rendering loop
      this.sprite = null;

      return this;
    },

    onAddToBlock: function( webglBlock ) {
      this.webglBlock = webglBlock; // TODO: do we need this reference?
      this.markDirty();

      this.reserveSprite();
    },

    onRemoveFromBlock: function( webglBlock ) {
      this.unreserveSprite();
    },

    reserveSprite: function() {
      if ( this.sprite ) {
        // if we already reserved a sprite for the image, bail out
        if ( this.sprite.image === this.node._image ) {
          return;
        }
        // otherwise we need to ditch our last reservation before reserving a new sprite
        else {
          this.unreserveSprite();
        }
      }

      // if the width/height isn't loaded yet, we can still use the desired value
      var width = this.node.getImageWidth();
      var height = this.node.getImageHeight();

      // if we have a width/height, we'll load a sprite
      this.sprite = ( width > 0 && height > 0 ) ? this.webglBlock.addSpriteSheetImage( this.node._image, width, height ) : null;

      // full updates on everything if our sprite changes
      this.xyDirty = true;
      this.uvDirty = true;
    },

    unreserveSprite: function() {
      if ( this.sprite ) {
        this.webglBlock.removeSpriteSheetImage( this.sprite );
      }
      this.sprite = null;
    },

    // @override
    markTransformDirty: function() {
      this.xyDirty = true;

      WebGLSelfDrawable.prototype.markTransformDirty.call( this );
    },

    /**
     * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
     * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
     * @public (scenery-internal)
     *
     * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
     * dirty.
     */
    markPaintDirty: function() {
      this.xyDirty = true; // vertex positions can depend on image width/height
      this.uvDirty = true;

      this.markDirty();
    },

    update: function() {
      if ( this.dirty ) {
        this.dirty = false;

        // ensure that we have a reserved sprite (part of the spritesheet)
        this.reserveSprite();

        if ( this.dirtyImageOpacity || !this.updatedOnce ) {
          this.vertexArray[ VERTEX_0_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
          this.vertexArray[ VERTEX_1_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
          this.vertexArray[ VERTEX_2_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
          this.vertexArray[ VERTEX_3_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
          this.vertexArray[ VERTEX_4_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
          this.vertexArray[ VERTEX_5_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
        }
        this.updatedOnce = true;

        // if we don't have a sprite (we don't have a loaded image yet), just bail
        if ( !this.sprite ) {
          return;
        }

        if ( this.uvDirty ) {
          this.uvDirty = false;

          var uvBounds = this.sprite.uvBounds;

          // TODO: consider reversal of minY and maxY usage here for vertical inverse

          // first triangle UVs
          this.vertexArray[ VERTEX_0_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // upper left U
          this.vertexArray[ VERTEX_0_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper left V
          this.vertexArray[ VERTEX_1_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // lower left U
          this.vertexArray[ VERTEX_1_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower left V
          this.vertexArray[ VERTEX_2_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // upper right U
          this.vertexArray[ VERTEX_2_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper right V

          // second triangle UVs
          this.vertexArray[ VERTEX_3_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // upper right U
          this.vertexArray[ VERTEX_3_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper right V
          this.vertexArray[ VERTEX_4_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // lower left U
          this.vertexArray[ VERTEX_4_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower left V
          this.vertexArray[ VERTEX_5_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // lower right U
          this.vertexArray[ VERTEX_5_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower right V
        }

        if ( this.xyDirty ) {
          this.xyDirty = false;

          var width = this.node.getImageWidth();
          var height = this.node.getImageHeight();

          var transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
          transformMatrix.multiplyVector2( this.upperLeft.setXY( 0, 0 ) );
          transformMatrix.multiplyVector2( this.lowerLeft.setXY( 0, height ) );
          transformMatrix.multiplyVector2( this.upperRight.setXY( width, 0 ) );
          transformMatrix.multiplyVector2( this.lowerRight.setXY( width, height ) );

          // first triangle XYs
          this.vertexArray[ VERTEX_0_OFFSET + VERTEX_X_OFFSET ] = this.upperLeft.x;
          this.vertexArray[ VERTEX_0_OFFSET + VERTEX_Y_OFFSET ] = this.upperLeft.y;
          this.vertexArray[ VERTEX_1_OFFSET + VERTEX_X_OFFSET ] = this.lowerLeft.x;
          this.vertexArray[ VERTEX_1_OFFSET + VERTEX_Y_OFFSET ] = this.lowerLeft.y;
          this.vertexArray[ VERTEX_2_OFFSET + VERTEX_X_OFFSET ] = this.upperRight.x;
          this.vertexArray[ VERTEX_2_OFFSET + VERTEX_Y_OFFSET ] = this.upperRight.y;

          // second triangle XYs
          this.vertexArray[ VERTEX_3_OFFSET + VERTEX_X_OFFSET ] = this.upperRight.x;
          this.vertexArray[ VERTEX_3_OFFSET + VERTEX_Y_OFFSET ] = this.upperRight.y;
          this.vertexArray[ VERTEX_4_OFFSET + VERTEX_X_OFFSET ] = this.lowerLeft.x;
          this.vertexArray[ VERTEX_4_OFFSET + VERTEX_Y_OFFSET ] = this.lowerLeft.y;
          this.vertexArray[ VERTEX_5_OFFSET + VERTEX_X_OFFSET ] = this.lowerRight.x;
          this.vertexArray[ VERTEX_5_OFFSET + VERTEX_Y_OFFSET ] = this.lowerRight.y;
        }
      }
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      // TODO: disposal of buffers?

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageWebGLDrawable );
  // This sets up ImageWebGLDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Image.ImageWebGLDrawable );

  return Image;
} );


