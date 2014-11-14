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
  var WebGLBlock = require( 'SCENERY/display/WebGLBlock' );

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
    },

    getImage: function() {
      return this._image;
    },

    invalidateSupportedRenderers: function() {
      if ( this._image instanceof HTMLCanvasElement ) {
        this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsWebGL );
      }
      else {
        // assumes HTMLImageElement
        this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskSupportsDOM | scenery.bitmaskSupportsWebGL );
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

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[i].markDirtyImage();
        }

        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },

    getImageWidth: function() {
      return this._image.naturalWidth || this._image.width;
    },

    getImageHeight: function() {
      return this._image.naturalHeight || this._image.height;
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

    set image( value ) { this.setImage( value ); },
    get image() { return this.getImage(); },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + ( this._image.src ? this._image.src.replace( /'/g, '\\\'' ) : 'other' ) + '\', {' + propLines + '} )';
    }
  } );

  Image.prototype._mutatorKeys = [ 'image' ].concat( Node.prototype._mutatorKeys );

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
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  var ImageRenderState = Image.ImageRenderState = function( drawableType ) {
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

  /* jshint -W064 */
  ImageRenderState( ImageDOMDrawable );

  /* jshint -W064 */
  SelfDrawable.Poolable( ImageDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Image.ImageSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function ImageSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: ImageRenderState,
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
    dirtyMethods: ['markDirtyImage']
  } );


  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Image.ImageWebGLDrawable = inherit( WebGLSelfDrawable, function ImageWebGLDrawable( renderer, instance ) {
    this.initializeWebGLSelfDrawable( renderer, instance );

    //Small triangle strip that creates a square, which will be transformed into the right rectangle shape
    this.vertexCoordinates = new Float32Array( [
      0, 0,
      1, 0,
      0, 1,
      1, 1
    ] );
  }, {
    initializeContext: function( gl ) {
      // cleanup old buffer, if applicable
      this.disposeWebGLBuffers();

      this.gl = gl;
      this.buffer = gl.createBuffer();
//      this.initializePaintableState();
      this.updateRectangle();
    },

    // methods for forwarding dirty messages
    canvasSelfDirty: function() {
      // we pass this method and it is only called with blah.call( ... ), where the 'this' reference is set. ignore jshint
      /* jshint -W040 */
      this.markDirty();
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateRectangle: function() {
      var gl = this.gl;

      var rect = this.node;
      rect._rectX = 10;
      rect._rectY = 10;
      rect._rectWidth = 10;
      rect._rectHeight = 10;

      this.vertexCoordinates[0] = rect._rectX;
      this.vertexCoordinates[1] = rect._rectY;

      this.vertexCoordinates[2] = rect._rectX + rect._rectWidth;
      this.vertexCoordinates[3] = rect._rectY;

      this.vertexCoordinates[4] = rect._rectX;
      this.vertexCoordinates[5] = rect._rectY + rect._rectHeight;

      this.vertexCoordinates[6] = rect._rectX + rect._rectWidth;
      this.vertexCoordinates[7] = rect._rectY + rect._rectHeight;

      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
      gl.bufferData(
        gl.ARRAY_BUFFER,

        this.vertexCoordinates,

        //TODO: Once we are lazily handling the full matrix, we may benefit from DYNAMIC draw here, and updating the vertices themselves
        gl.STATIC_DRAW );

      // TODO: What to do when image changed and marked as dirty?  Where does that happen?
      if ( this.dirtyFill || true ) {
//        this.cleanPaintableState();
      }
    },

    render: function( shaderProgram ) {
      debugger;
      var gl = this.gl;

      // TODO: Handle rounded rectangles, please!
      // use the standard version if it's a rounded rectangle, since there is no WebGL-optimized version for that
      // TODO: how to handle fill/stroke delay optimizations here?
      //OHTWO TODO: optimize

      //TODO: what if image is null?
      var viewMatrix = this.instance.relativeMatrix.toAffineMatrix4();

      // combine image matrix (to scale aspect ratios), the trail's matrix, and the matrix to device coordinates
      gl.uniformMatrix4fv( shaderProgram.uniformLocations.uModelViewMatrix, false, viewMatrix.entries );

      //Indicate the branch of logic to use in the ubershader.  In this case, a texture should be used for the image
      gl.uniform1i( shaderProgram.uniformLocations.uFragmentType, WebGLBlock.fragmentTypeFill );
      gl.uniform4f( shaderProgram.uniformLocations.uColor, 1, 0, 0, 1 );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
      gl.vertexAttribPointer( shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 0, 0 );
      gl.drawArrays( gl.TRIANGLE_STRIP, 0, 4 );
    },

    dispose: function() {
      // super
      WebGLSelfDrawable.prototype.dispose.call( this );

      this.disposeWebGLBuffers();
    },

    disposeWebGLBuffers: function() {
      if ( this.gl ) {
        this.gl.deleteBuffer( this.buffer );
      }
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

    update: function() {
      this.dirty = false;

      if ( this.paintDirty ) {
        this.updateRectangle();

        this.setToCleanState();
      }
    }
  } );

  // set up pooling
  /* jshint -W064 */
  SelfDrawable.Poolable( Image.ImageWebGLDrawable );

  return Image;
} );


