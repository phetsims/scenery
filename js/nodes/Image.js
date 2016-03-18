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
   *
   * initialHeight - {number} If the input image hasn't loaded yet, but the (expected) size is known, providing an
   *    initialHeight will cause the Image node to have the correct bounds (height) before the pixel data has been fully
   *    loaded. A value of 0 will be ignored.
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
    },

    getImage: function() {
      return this._image;
    },
    get image() { return this.getImage(); },

    invalidateSupportedRenderers: function() {
      if ( this._image instanceof HTMLCanvasElement ) {
        this.setRendererBitmask(
          Renderer.bitmaskCanvas |
          Renderer.bitmaskWebGL
        );
      }
      else {
        // assumes HTMLImageElement
        this.setRendererBitmask(
          Renderer.bitmaskCanvas |
          Renderer.bitmaskSVG |
          Renderer.bitmaskDOM |
          Renderer.bitmaskWebGL
        );
      }
    },

    setImage: function( image ) {
      if ( this._image !== image && ( typeof image !== 'string' || !this._image || ( image !== this._image.src && image !== this._mipmapData ) ) ) {
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

        // swap supported renderers if necessary
        this.invalidateSupportedRenderers();

        this._image = image;

        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },
    set image( value ) { this.setImage( value ); },

    getInitialWidth: function() {
      return this._initialWidth;
    },
    get initialWidth() { return this.getInitialWidth(); },

    setInitialWidth: function( width ) {
      this._initialWidth = width;

      this.invalidateImage();
    },
    set initialWidth( value ) { this.setInitialWidth( value ); },

    getInitialHeight: function() {
      return this._initialHeight;
    },
    get initialHeight() { return this.getInitialHeight(); },

    setInitialHeight: function( height ) {
      this._initialHeight = height;

      this.invalidateImage();
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

    // signal that we are actually rendering something
    isPainted: function() {
      return true;
    },

    canvasPaintSelf: function( wrapper ) {
      Image.ImageCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createDOMDrawable: function( renderer, instance ) {
      return Image.ImageDOMDrawable.createFromPool( renderer, instance );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Image.ImageSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Image.ImageCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Image.ImageWebGLDrawable.createFromPool( renderer, instance );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + ( this._image.src ? this._image.src.replace( /'/g, '\\\'' ) : 'other' ) + '\', {' + propLines + '} )';
    }
  } );

  Image.prototype._mutatorKeys = [ 'image', 'initialWidth', 'initialHeight', 'mipmap', 'mipmapBias', 'mipmapInitialLevel', 'mipmapMaxLevel' ].concat( Node.prototype._mutatorKeys );

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

  // Creates an {object} suitable to be passed to Image as a mipmap (from a Canvas)
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

  Image.ImageStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function( renderer, instance ) {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyImage = true;
        this.dirtyMipmap = true;

        return this; // allow for chaining
      };

      proto.disposeState = function() {

      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyImage = function() {
        this.dirtyImage = true;
        this.markPaintDirty();
      };

      proto.markDirtyMipmap = function() {
        this.dirtyMipmap = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyImage = false;
        this.dirtyMipmap = false;
      };
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  Image.ImageDOMDrawable = inherit( DOMSelfDrawable, function ImageDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // initializes, and resets (so we can support pooled states)
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
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

      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    updateDOM: function() {
      var node = this.node;
      var img = this.domElement;

      if ( this.paintDirty && this.dirtyImage ) {
        // TODO: allow other ways of showing a DOM image?
        img.src = node._image ? node._image.src : '//:0'; // NOTE: for img with no src (but with a string), see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
      }

      if ( this.transformDirty ) {
        scenery.Util.applyPreparedTransform( this.getTransformMatrix(), this.domElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToClean();
    },

    setToClean: function() {
      this.setToCleanState();

      this.transformDirty = false;
    },

    dispose: function() {
      this.disposeState();

      if ( !keepDOMImageElements ) {
        this.domElement = null; // clear our DOM reference if we want to toss it
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageDOMDrawable );
  SelfDrawable.Poolable.mixin( Image.ImageDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Image.ImageSVGDrawable = function ImageSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Image.ImageSVGDrawable, {
    initialize: function( renderer, instance ) {
      this.initializeSVGSelfDrawable( renderer, instance, false, keepSVGImageElements ); // usesPaint: false

      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' initialized for ' + instance.toString() );
      var self = this;

      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'image' );
        this.svgElement.setAttribute( 'x', 0 );
        this.svgElement.setAttribute( 'y', 0 );
      }

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

    dispose: function() {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( this.id + ' disposing' );

      // clean up mipmap listeners and compute needs
      this.updateMipmapStatus( false );
      this.node.offStatic( 'mipmap', this._mipmapListener );

      SVGSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageSVGDrawable );
  SelfDrawable.Poolable.mixin( Image.ImageSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Image.ImageCanvasDrawable = function ImageCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Image.ImageCanvasDrawable, {
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    paintCanvas: function( wrapper, node ) {
      // Ensure that the image has been loaded by checking whether it has a width or height of 0.
      // See https://github.com/phetsims/scenery/issues/536
      if ( node._image && node._image.width !== 0 && node._image.height !== 0 ) {
        wrapper.context.drawImage( node._image, 0, 0 );
      }
    },

    // stateless dirty functions
    markDirtyImage: function() { this.markPaintDirty(); },
    markDirtyMipmap: function() { this.markPaintDirty(); }
  } );
  SelfDrawable.Poolable.mixin( Image.ImageCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Image.ImageWebGLDrawable = inherit( WebGLSelfDrawable, function ImageWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    webglRenderer: Renderer.webglTexturedTriangles,

    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );

      if ( !this.vertexArray ) {
        // format [X Y U V] for 6 vertices
        this.vertexArray = new Float32Array( 4 * 6 ); // 4-length components for 6 vertices (2 tris).
      }

      // corner vertices in the relative transform root coordinate space
      this.upperLeft = new Vector2();
      this.lowerLeft = new Vector2();
      this.upperRight = new Vector2();
      this.lowerRight = new Vector2();

      this.xyDirty = true; // is our vertex position information out of date?
      this.uvDirty = true; // is our UV information out of date?

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

    // called when something about the Image's image itself changes (not transform, etc.)
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

        // if we don't have a sprite (we don't have a loaded image yet), just bail
        if ( !this.sprite ) {
          return;
        }

        if ( this.uvDirty ) {
          this.uvDirty = false;

          var uvBounds = this.sprite.uvBounds;

          // TODO: consider reversal of minY and maxY usage here for vertical inverse

          // first triangle UVs
          this.vertexArray[ 2 ] = uvBounds.minX; // upper left U
          this.vertexArray[ 3 ] = uvBounds.minY; // upper left V
          this.vertexArray[ 6 ] = uvBounds.minX; // lower left U
          this.vertexArray[ 7 ] = uvBounds.maxY; // lower left V
          this.vertexArray[ 10 ] = uvBounds.maxX; // upper right U
          this.vertexArray[ 11 ] = uvBounds.minY; // upper right V

          // second triangle UVs
          this.vertexArray[ 14 ] = uvBounds.maxX; // upper right U
          this.vertexArray[ 15 ] = uvBounds.minY; // upper right V
          this.vertexArray[ 18 ] = uvBounds.minX; // lower left U
          this.vertexArray[ 19 ] = uvBounds.maxY; // lower left V
          this.vertexArray[ 22 ] = uvBounds.maxX; // lower right U
          this.vertexArray[ 23 ] = uvBounds.maxY; // lower right V
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
          this.vertexArray[ 0 ] = this.upperLeft.x;
          this.vertexArray[ 1 ] = this.upperLeft.y;
          this.vertexArray[ 4 ] = this.lowerLeft.x;
          this.vertexArray[ 5 ] = this.lowerLeft.y;
          this.vertexArray[ 8 ] = this.upperRight.x;
          this.vertexArray[ 9 ] = this.upperRight.y;

          // second triangle XYs
          this.vertexArray[ 12 ] = this.upperRight.x;
          this.vertexArray[ 13 ] = this.upperRight.y;
          this.vertexArray[ 16 ] = this.lowerLeft.x;
          this.vertexArray[ 17 ] = this.lowerLeft.y;
          this.vertexArray[ 20 ] = this.lowerRight.x;
          this.vertexArray[ 21 ] = this.lowerRight.y;
        }
      }
    },

    dispose: function() {
      // TODO: disposal of buffers?

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Image.ImageStatefulDrawable.mixin( Image.ImageWebGLDrawable );
  SelfDrawable.Poolable.mixin( Image.ImageWebGLDrawable ); // pooling

  return Image;
} );


