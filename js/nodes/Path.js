// Copyright 2002-2014, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Shape = require( 'KITE/Shape' );

  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/display/Renderer' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var WebGLBlock = require( 'SCENERY/display/WebGLBlock' );
  var Util = require( 'SCENERY/util/Util' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  scenery.Path = function Path( shape, options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    // NOTE: _shape can be lazily constructed, in the case of types like Rectangle where they have their own drawing code
    this._shape = null;
    this._strokedShape = null; // a stroked copy of the shape, lazily computed

    // ensure we have a parameter object
    options = options || {};

    this.initializePaintable();

    Node.call( this );
    this.invalidateSupportedRenderers();
    this.setShape( shape );
    this.mutate( options );
  };
  var Path = scenery.Path;

  inherit( Node, Path, {
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getPathRendererBitmask: function() {
      return scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskSupportsWebGL;
    },

    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
    },

    // sets the shape drawn, or null to remove the shape
    setShape: function( shape ) {
      if ( this._shape !== shape ) {
        if ( typeof shape === 'string' ) {
          // be content with setShape always invalidating the shape?
          shape = new Shape( shape );
        }
        this._shape = shape;
        this.invalidateShape();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[i].markDirtyShape();
        }
      }
      return this;
    },

    getShape: function() {
      return this._shape;
    },

    getStrokedShape: function() {
      if ( !this._strokedShape ) {
        this._strokedShape = this.getShape().getStrokedShape( this._lineDrawingStyles );
      }
      return this._strokedShape;
    },

    invalidateShape: function() {
      this._strokedShape = null;

      if ( this.hasShape() ) {
        this.invalidateSelf( this.computeShapeBounds() );
      }
    },

    // separated out, so that we can override this with a faster version in subtypes. includes the Stroke, if any
    computeShapeBounds: function() {
      return this._stroke ? this.getStrokedShape().bounds : this.getShape().bounds;
    },

    // @override
    getTransformedSelfBounds: function( matrix ) {
      return ( this._stroke ? this.getStrokedShape() : this.getShape() ).getBoundsWithTransform( matrix );
    },

    // hook stroke mixin changes to invalidation
    invalidateStroke: function() {
      this.invalidateShape();
    },

    hasShape: function() {
      return this._shape;
    },

    canvasPaintSelf: function( wrapper ) {
      Path.PathCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Path.PathSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Path.PathCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Path.PathWebGLDrawable.createFromPool( renderer, instance );
    },

    isPainted: function() {
      return true;
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      var result = false;
      if ( !this.hasShape() ) {
        return result;
      }

      // if this node is fillPickable, we will return true if the point is inside our fill area
      if ( this._fillPickable ) {
        result = this.getShape().containsPoint( point );
      }

      // also include the stroked region in the hit area if strokePickable
      if ( !result && this._strokePickable ) {
        result = this.getStrokedShape().containsPoint( point );
      }
      return result;
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // TODO: should a shape's stroke be included?
      return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
    },

    set shape( value ) { this.setShape( value ); },
    get shape() { return this.getShape(); },

    getDebugHTMLExtras: function() {
      return this._shape ? ' (<span style="color: #88f" onclick="window.open( \'data:text/plain;charset=utf-8,\' + encodeURIComponent( \'' + this._shape.getSVGPath() + '\' ) );">path</span>)' : '';
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Path( ' + ( this._shape ? this._shape.toString() : this._shape ) + ', {' + propLines + '} )';
    },

    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      return result;
    }
  } );

  Path.prototype._mutatorKeys = [ 'shape' ].concat( Node.prototype._mutatorKeys );

  // mix in fill/stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  /* jshint -W064 */
  Paintable( Path );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  var PathStatefulDrawableMixin = Path.PathStatefulDrawableMixin = function( drawableType ) {
    var proto = drawableType.prototype;

    // initializes, and resets (so we can support pooled states)
    proto.initializeState = function() {
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyShape = true;

      // adds fill/stroke-specific flags and state
      this.initializePaintableState();

      return this; // allow for chaining
    };

    // catch-all dirty, if anything that isn't a transform is marked as dirty
    proto.markPaintDirty = function() {
      this.paintDirty = true;
      this.markDirty();
    };

    proto.markDirtyShape = function() {
      this.dirtyShape = true;
      this.markPaintDirty();
    };

    proto.setToCleanState = function() {
      this.paintDirty = false;
      this.dirtyShape = false;

      this.cleanPaintableState();
    };

    /* jshint -W064 */
    Paintable.PaintableStatefulDrawableMixin( drawableType );
  };

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Path.PathSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function PathSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: PathStatefulDrawableMixin,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'path' );
      }
    },
    updateSVG: function( node, path ) {
      assert && assert( !node.requiresSVGBoundsWorkaround(), 'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );

      if ( this.dirtyShape ) {
        var svgPath = node.hasShape() ? node._shape.getSVGPath() : '';

        // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
        // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
        // the attribute can cause this bug
        if ( !svgPath ) { svgPath = 'M0 0'; }

        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }

      this.updateFillStrokeStyle( path );
    },
    usesPaint: true,
    keepElements: keepSVGPathElements
  } );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Path.PathCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function PathCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasPath( wrapper, node ) {
      var context = wrapper.context;

      if ( node.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        node._shape.writeToContext( context );

        if ( node._fill ) {
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fill();
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }
        if ( node._stroke ) {
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.stroke();
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
      }
    },
    usesPaint: true,
    dirtyMethods: ['markDirtyShape']
  } );


  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Path.PathWebGLDrawable = inherit( WebGLSelfDrawable, function PathWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called from the constructor OR from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );

      //Small triangle strip that creates a square, which will be transformed into the right rectangle shape
      this.vertexCoordinates = this.vertexCoordinates || new Float32Array( 8 );

      this.textureCoordinates = this.textureCoordinates || new Float32Array( [
        0, 0,
        1, 0,
        0, 1,
        1, 1
      ] );
    },

    initializeContext: function( gl ) {
      assert && assert( gl );

      this.gl = gl;

      // cleanup old buffer, if applicable
      this.disposeWebGLBuffers();

      // holds vertex coordinates
      this.vertexBuffer = gl.createBuffer();

      // holds texture U,V coordinate pairs pointing into our texture coordinate space
      this.textureBuffer = gl.createBuffer();

      this.updateImage();
    },

    transformVertexCoordinateX: function( x ) {
      return x * this.canvasWidth + this.cachedBounds.minX;
    },

    transformVertexCoordinateY: function( y ) {
      return ( 1 - y ) * this.canvasHeight + this.cachedBounds.minY;
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateImage: function() {
      var gl = this.gl;

      if ( this.texture !== null ) {
        gl.deleteTexture( this.texture );
      }

      if ( this.node._shape ) {
        // TODO: only create once instance of this Canvas for reuse
        var canvas = document.createElement( 'canvas' );
        var context = canvas.getContext( '2d' );

        this.cachedBounds = this.node.getShape().bounds;
        console.log( this.cachedBounds );

        // TODO: Account for stroke
        this.canvasWidth = canvas.width = Util.toPowerOf2( this.cachedBounds.width );
        this.canvasHeight = canvas.height = Util.toPowerOf2( this.cachedBounds.height );
        var image = this.node.toCanvasNodeSynchronous().children[0].image;
        context.drawImage( image, 0, 0 );

        var texture = this.texture = gl.createTexture();
        gl.bindTexture( gl.TEXTURE_2D, texture );
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );

        gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, true );
        gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas );

        // Texture filtering, see http://learningwebgl.com/blog/?p=571
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
        gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST );
        gl.generateMipmap( gl.TEXTURE_2D );

        gl.bindTexture( gl.TEXTURE_2D, null );

        this.vertexCoordinates[0] = this.transformVertexCoordinateX( 0 );
        this.vertexCoordinates[1] = this.transformVertexCoordinateY( 0 );

        this.vertexCoordinates[2] = this.transformVertexCoordinateX( 1 );
        this.vertexCoordinates[3] = this.transformVertexCoordinateY( 0 );

        this.vertexCoordinates[4] = this.transformVertexCoordinateX( 0 );
        this.vertexCoordinates[5] = this.transformVertexCoordinateY( 1 );

        this.vertexCoordinates[6] = this.transformVertexCoordinateX( 1 );
        this.vertexCoordinates[7] = this.transformVertexCoordinateY( 1 );

        gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

        //TODO: Once we are lazily handling the full matrix, we may benefit from DYNAMIC draw here, and updating the vertices themselves
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexCoordinates, gl.STATIC_DRAW );

        gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
        gl.bufferData( gl.ARRAY_BUFFER, this.textureCoordinates, gl.STATIC_DRAW );
      }
    },

    render: function( shaderProgram ) {
      if ( this.node._shape ) {
        var gl = this.gl;

        //TODO: what if image is null?

        //OHTWO TODO: optimize
        //TODO: This looks like an expense we don't want to incur at every render.  How about moving it to the GPU?
        var viewMatrix = this.instance.relativeTransform.matrix.toAffineMatrix4();

        // combine image matrix (to scale aspect ratios), the trail's matrix, and the matrix to device coordinates
        gl.uniformMatrix4fv( shaderProgram.uniformLocations.uModelViewMatrix, false, viewMatrix.entries );

        gl.uniform1i( shaderProgram.uniformLocations.uTexture, 0 ); // TEXTURE0 slot

        //Indicate the branch of logic to use in the ubershader.  In this case, a texture should be used for the image
        gl.uniform1i( shaderProgram.uniformLocations.uFragmentType, WebGLBlock.fragmentTypeTexture );

        gl.activeTexture( gl.TEXTURE0 );
        gl.bindTexture( gl.TEXTURE_2D, this.texture );

        gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
        gl.vertexAttribPointer( shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 0, 0 );

        gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
        gl.vertexAttribPointer( shaderProgram.attributeLocations.aTexCoord, 2, gl.FLOAT, false, 0, 0 );

        gl.drawArrays( gl.TRIANGLE_STRIP, 0, 4 );
      }
    },

    shaderAttributes: [
      'aVertex',
      'aTexCoord'
    ],

    dispose: function() {
      // we may have been disposed without initializeContext being called (never attached to a block)
      if ( this.gl ) {
        this.disposeWebGLBuffers();
        this.gl = null;
      }

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );

    },

    disposeWebGLBuffers: function() {
      this.gl.deleteBuffer( this.vertexBuffer );
      this.gl.deleteBuffer( this.textureBuffer );
      this.gl.deleteTexture( this.texture );
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
      if ( this.dirtyShape ) {
        this.updateImage();

        this.setToCleanState();
      }

      this.dirty = false;
    }
  } );

  // set up pooling
  /* jshint -W064 */
  SelfDrawable.PoolableMixin( Path.PathWebGLDrawable );

  /* jshint -W064 */
  PathStatefulDrawableMixin( Path.PathWebGLDrawable );


  return Path;
} );


