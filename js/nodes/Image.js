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
  var scenery = require( 'SCENERY/scenery' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var SpriteSheet = require( 'SCENERY/util/SpriteSheet' );
  var ImageCanvasDrawable = require( 'SCENERY/display/drawables/ImageCanvasDrawable' );
  var ImageDOMDrawable = require( 'SCENERY/display/drawables/ImageDOMDrawable' );
  var ImageSVGDrawable = require( 'SCENERY/display/drawables/ImageSVGDrawable' );
  var ImageWebGLDrawable = require( 'SCENERY/display/drawables/ImageWebGLDrawable' );

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
      ImageCanvasDrawable.prototype.paintCanvas( wrapper, this );
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

  return Image;
} );
