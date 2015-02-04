// Copyright 2002-2014, University of Colorado Boulder

/**
 * Images
 *
 * TODO: allow multiple DOM instances (create new HTMLImageElement elements)
 * TODO: SVG support
 * TODO: support rendering a Canvas to DOM (single instance)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  require( 'SCENERY/display/Renderer' ); // we need to specify the Renderer in the prototype
  require( 'SCENERY/util/Util' );

  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var PixiSelfDrawable = require( 'SCENERY/display/PixiSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMImageElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGImageElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /*
   * Canvas renderer supports the following as 'image':
   *     URL (string)             // works, but does NOT support bounds-based parameter object keys like 'left', 'centerX', etc.
   *                              // also necessary to force updateScene() after it has loaded
   *     HTMLImageElement         // works
   *     HTMLVideoElement         // not tested
   *     HTMLCanvasElement        // works, and forces the canvas renderer
   *     CanvasRenderingContext2D // not tested, but bad luck in past
   *     ImageBitmap              // good luck creating this. currently API for window.createImageBitmap not implemented
   * SVG renderer supports the following as 'image':
   *     URL (string)
   *     HTMLImageElement
   */
  scenery.Image = function Image( image, options ) {
    assert && assert( image, "image should be available" );

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

    var self = this;
    // allows us to invalidate our bounds whenever an image is loaded
    this.loadListener = function( event ) {
      self.invalidateImage();

      // don't leak memory!
      self._image.removeEventListener( 'load', self.loadListener );
    };

    Node.call( this, options );
    this.invalidateSupportedRenderers();
  };
  var Image = scenery.Image;

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
    },

    getImage: function() {
      return this._image;
    },

    invalidateSupportedRenderers: function() {
      if ( this._image instanceof HTMLCanvasElement ) {
        this.setRendererBitmask(
          scenery.bitmaskBoundsValid |
          scenery.bitmaskSupportsCanvas |
          scenery.bitmaskSupportsWebGL |
          scenery.bitmaskSupportsPixi
        );
      }
      else {
        // assumes HTMLImageElement
        this.setRendererBitmask(
          scenery.bitmaskBoundsValid |
          scenery.bitmaskSupportsCanvas |
          scenery.bitmaskSupportsSVG |
          scenery.bitmaskSupportsDOM |
          scenery.bitmaskSupportsPixi
        );
      }
    },

    setImage: function( image ) {
      if ( this._image !== image && ( typeof image !== 'string' || !this._image || image !== this._image.src ) ) {
        // don't leak memory by referencing old images
        if ( this._image ) {
          this._image.removeEventListener( 'load', this.loadListener );
        }

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

        // swap supported renderers if necessary
        this.invalidateSupportedRenderers();

        this._image = image;

        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },

    getInitialWidth: function() {
      return this._initialWidth;
    },

    setInitialWidth: function( width ) {
      this._initialWidth = width;

      this.invalidateImage();
    },

    getInitialHeight: function() {
      return this._initialHeight;
    },

    setInitialHeight: function( height ) {
      this._initialHeight = height;

      this.invalidateImage();
    },

    getImageWidth: function() {
      var detectedWidth = this._image.naturalWidth || this._image.width;
      if ( detectedWidth === 0 ) {
        return this._initialWidth; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialWidth === 0 || this._initialWidth === detectedWidth, 'Bad Image.initialWidth' );

        return detectedWidth;
      }
    },

    getImageHeight: function() {
      var detectedHeight = this._image.naturalHeight || this._image.height;
      if ( detectedHeight === 0 ) {
        return this._initialHeight; // either 0 (default), or the overridden value
      }
      else {
        assert && assert( this._initialHeight === 0 || this._initialHeight === detectedHeight, 'Bad Image.initialHeight' );

        return detectedHeight;
      }
    },

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

    createPixiDrawable: function( renderer, instance ) {
      return Image.ImagePixiDrawable.createFromPool( renderer, instance );
    },

    set image( value ) { this.setImage( value ); },
    get image() { return this.getImage(); },

    set initialWidth( value ) { this.setInitialWidth( value ); },
    get initialWidth() { return this.getInitialWidth(); },

    set initialHeight( value ) { this.setInitialHeight( value ); },
    get initialHeight() { return this.getInitialHeight(); },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + ( this._image.src ? this._image.src.replace( /'/g, '\\\'' ) : 'other' ) + '\', {' + propLines + '} )';
    }
  } );

  Image.prototype._mutatorKeys = [ 'image', 'initialWidth', 'initialHeight' ].concat( Node.prototype._mutatorKeys );

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

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG) //TODO: Does this also apply to WebGL?
   *----------------------------------------------------------------------------*/

  Image.ImageStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyImage = true;

        return this; // allow for chaining
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

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyImage = false;
      };
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  var ImageDOMDrawable = Image.ImageDOMDrawable = inherit( DOMSelfDrawable, function ImageDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // initializes, and resets (so we can support pooled states)
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState();

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

    onAttach: function( node ) {

    },

    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function( node ) {
      if ( !keepDOMImageElements ) {
        // clear the references
        this.domElement = null;
      }
    },

    setToClean: function() {
      this.setToCleanState();

      this.transformDirty = false;
    }
  } );

  Image.ImageStatefulDrawable.mixin( ImageDOMDrawable );

  SelfDrawable.Poolable.mixin( ImageDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Image.ImageSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function ImageSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: Image.ImageStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'image' );
        this.svgElement.setAttribute( 'x', 0 );
        this.svgElement.setAttribute( 'y', 0 );
      }
    },
    updateSVG: function( node, image ) {
      //OHTWO TODO: performance: consider using <use> with <defs> for our image element. This could be a significant speedup!
      if ( this.dirtyImage ) {
        if ( node._image ) {
          // like <image xlink:href='http://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
          image.setAttribute( 'width', node.getImageWidth() + 'px' );
          image.setAttribute( 'height', node.getImageHeight() + 'px' );
          image.setAttributeNS( scenery.xlinkns, 'xlink:href', node.getImageURL() );
        }
        else {
          image.setAttribute( 'width', '0' );
          image.setAttribute( 'height', '0' );
          image.setAttributeNS( scenery.xlinkns, 'xlink:href', '//:0' ); // see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
        }
      }
    },
    usesPaint: false,
    keepElements: keepSVGImageElements
  } );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Image.ImageCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function ImageCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasImage( wrapper, node ) {
      if ( node._image ) {
        wrapper.context.drawImage( node._image, 0, 0 );
      }
    },
    usesPaint: false,
    dirtyMethods: [ 'markDirtyImage' ]
  } );


  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Image.ImageWebGLDrawable = inherit( WebGLSelfDrawable, function ImageWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    initializeContext: function( webglBlock ) {
      this.webglBlock = webglBlock;
      this.rectangleHandle = webglBlock.webglRenderer.textureRenderer.textureBufferData.createFromImageNode( this.node, 0.5 );

      // TODO: Don't call this each time a new item is added.
      webglBlock.webglRenderer.textureRenderer.bindVertexBuffer();
      webglBlock.webglRenderer.textureRenderer.bindDirtyTextures();
      // cleanup old vertexBuffer, if applicable
//      this.disposeWebGLBuffers();

      this.updateRectangle();

      //TODO: Update the state in the buffer arrays
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateRectangle: function() {
    },

    render: function( shaderProgram ) {
      // This is handled by the ColorTriangleRenderer
    },

    dispose: function() {
      this.disposeWebGLBuffers();

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    disposeWebGLBuffers: function() {
      this.webglBlock.webglRenderer.colorTriangleRenderer.colorTriangleBufferData.dispose( this.rectangleHandle );
    },

    markDirtyRectangle: function() {
      this.markDirty();
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    onAttach: function( node ) {

    },

    // release the drawable
    onDetach: function( node ) {
      //OHTWO TODO: are we missing the disposal?
    },

    //TODO: Make sure all of the dirty flags make sense here.  Should we be using fillDirty, paintDirty, dirty, etc?
    update: function() {
      if ( this.dirty ) {
        this.updateRectangle();
        this.dirty = false;
      }
    }
  } );

  // set up pooling
  SelfDrawable.Poolable.mixin( Image.ImageWebGLDrawable );

  Image.ImageStatefulDrawable.mixin( Image.ImageWebGLDrawable );


  /*---------------------------------------------------------------------------*
   * Pixi Rendering
   *----------------------------------------------------------------------------*/

  Image.ImagePixiDrawable = PixiSelfDrawable.createDrawable( {
    type: function ImagePixiDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: Image.ImageStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      if ( !this.displayObject ) {
        var baseTexture = new PIXI.BaseTexture( this.node._image, PIXI.scaleModes.DEFAULT );
        var texture = new PIXI.Texture( baseTexture );
        this.displayObject = new PIXI.Sprite( texture );
      }
    },
    updatePixi: function( node, image ) {
      //OHTWO TODO: performance: consider using <use> with <defs> for our image element. This could be a significant speedup!
      if ( this.dirtyImage ) {
        if ( node._image ) {
          var baseTexture = new PIXI.BaseTexture( this.node._image, PIXI.scaleModes.DEFAULT );
          var texture = new PIXI.Texture( baseTexture );
          this.displayObject.setTexture( texture );
        }
        else {
          this.displayObject.setTexture( null );
        }
      }
    },
    usesPaint: false,
    keepElements: keepSVGImageElements
  } );

  // set up pooling
  SelfDrawable.Poolable.mixin( Image.ImagePixiDrawable );

  Image.ImageStatefulDrawable.mixin( Image.ImagePixiDrawable );

  return Image;
} );


