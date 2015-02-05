// Copyright 2002-2014, University of Colorado Boulder

/**
 * A circular node that inherits Path, and allows for optimized drawing,
 * and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var Color = require( 'SCENERY/util/Color' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Features = require( 'SCENERY/util/Features' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  require( 'SCENERY/util/Util' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var WebGLBlock = require( 'SCENERY/display/WebGLBlock' );
  var PixiSelfDrawable = require( 'SCENERY/display/PixiSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMCircleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGCircleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  scenery.Circle = function Circle( radius, options ) {
    if ( typeof radius === 'object' ) {
      // allow new Circle( { radius: ... } )
      // the mutators will call invalidateCircle() and properly set the shape
      options = radius;
      this._radius = options.radius;
    }
    else {
      this._radius = radius;

      // ensure we have a parameter object
      options = options || {};

    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  };
  var Circle = scenery.Circle;

  inherit( Path, Circle, {
    getStrokeRendererBitmask: function() {
      var bitmask = Path.prototype.getStrokeRendererBitmask.call( this );
      if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && this.getLineWidth() <= this.getRadius() ) {
        bitmask |= scenery.bitmaskSupportsDOM;
      }
      bitmask |= scenery.bitmaskSupportsPixi;
      return bitmask;
    },

    getPathRendererBitmask: function() {
      return scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskBoundsValid | scenery.bitmaskSupportsPixi | ( Features.borderRadius ? scenery.bitmaskSupportsDOM : 0 );
    },

    invalidateCircle: function() {
      assert && assert( this._radius >= 0, 'A circle needs a non-negative radius' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidateShape();
    },

    createCircleShape: function() {
      return Shape.circle( 0, 0, this._radius );
    },

    intersectsBoundsSelf: function( bounds ) {
      // TODO: handle intersection with somewhat-infinite bounds!
      var x = Math.abs( bounds.centerX );
      var y = Math.abs( bounds.centerY );
      var halfWidth = bounds.maxX - x;
      var halfHeight = bounds.maxY - y;

      // too far to have a possible intersection
      if ( x > halfWidth + this._radius || y > halfHeight + this._radius ) {
        return false;
      }

      // guaranteed intersection
      if ( x <= halfWidth || y <= halfHeight ) {
        return true;
      }

      // corner case
      x -= halfWidth;
      y -= halfHeight;
      return x * x + y * y <= this._radius * this._radius;
    },

    canvasPaintSelf: function( wrapper ) {
      Circle.CircleCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createDOMDrawable: function( renderer, instance ) {
      return Circle.CircleDOMDrawable.createFromPool( renderer, instance );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Circle.CircleSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Circle.CircleCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Circle.CircleWebGLDrawable.createFromPool( renderer, instance );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._radius + ', {' + propLines + '} )';
    },

    getRadius: function() {
      return this._radius;
    },

    setRadius: function( radius ) {
      assert && assert( typeof radius === 'number', 'Circle.radius must be a number' );

      if ( this._radius !== radius ) {
        this._radius = radius;
        this.invalidateCircle();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyRadius();
        }
      }
      return this;
    },

    computeShapeBounds: function() {
      var bounds = new Bounds2( -this._radius, -this._radius, this._radius, this._radius );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
    },

    // accelerated hit detection
    containsPointSelf: function( point ) {
      var magSq = point.x * point.x + point.y * point.y;
      var result = true;
      var iRadius;
      if ( this._strokePickable ) {
        iRadius = this.getLineWidth() / 2;
        var outerRadius = this._radius + iRadius;
        result = result && magSq <= outerRadius * outerRadius;
      }

      if ( this._fillPickable ) {
        if ( this._strokePickable ) {
          // we were either within the outer radius, or not
          return result;
        }
        else {
          // just testing in the fill range
          return magSq <= this._radius * this._radius;
        }
      }
      else if ( this._strokePickable ) {
        var innerRadius = this._radius - iRadius;
        return result && magSq >= innerRadius * innerRadius;
      }
      else {
        return false; // neither stroke nor fill is pickable
      }
    },

    get radius() { return this.getRadius(); },
    set radius( value ) { return this.setRadius( value ); },

    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Circle to something non-null' );
      }
      else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },

    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createCircleShape();
      }
      return this._shape;
    },

    hasShape: function() {
      return true;
    }
  } );

  // not adding mutators for now
  Circle.prototype._mutatorKeys = [ 'radius' ].concat( Path.prototype._mutatorKeys );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  Circle.CircleStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyRadius = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState();

        return this; // allow for chaining
      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyRadius = function() {
        this.dirtyRadius = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyRadius = false;

        this.cleanPaintableState();
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  var CircleDOMDrawable = Circle.CircleDOMDrawable = inherit( DOMSelfDrawable, function CircleDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // initializes, and resets (so we can support pooled states)
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState();

      if ( !this.matrix ) {
        this.matrix = Matrix3.dirtyFromPool();
      }

      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.fillElement || !this.strokeElement ) {
        var fillElement = this.fillElement = document.createElement( 'div' );
        var strokeElement = this.strokeElement = document.createElement( 'div' );
        fillElement.style.display = 'block';
        fillElement.style.position = 'absolute';
        fillElement.style.left = '0';
        fillElement.style.top = '0';
        fillElement.style.pointerEvents = 'none';
        strokeElement.style.display = 'block';
        strokeElement.style.position = 'absolute';
        strokeElement.style.left = '0';
        strokeElement.style.top = '0';
        strokeElement.style.pointerEvents = 'none';
        fillElement.appendChild( strokeElement );
      }

      this.domElement = this.fillElement;

      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    updateDOM: function() {
      var node = this.node;
      var fillElement = this.fillElement;
      var strokeElement = this.strokeElement;

      if ( this.paintDirty ) {
        if ( this.dirtyRadius ) {
          fillElement.style.width = ( 2 * node._radius ) + 'px';
          fillElement.style.height = ( 2 * node._radius ) + 'px';
          fillElement.style[ Features.borderRadius ] = node._radius + 'px';
        }
        if ( this.dirtyFill ) {
          fillElement.style.backgroundColor = node.getCSSFill();
        }

        if ( this.dirtyStroke ) {
          // update stroke presence
          if ( node.hasStroke() ) {
            strokeElement.style.borderStyle = 'solid';
          }
          else {
            strokeElement.style.borderStyle = 'none';
          }
        }

        if ( node.hasStroke() ) {
          // since we only execute these if we have a stroke, we need to redo everything if there was no stroke previously.
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only Circles)
          var hadNoStrokeBefore = this.lastStroke === null;

          if ( hadNoStrokeBefore || this.dirtyLineWidth || this.dirtyRadius ) {
            strokeElement.style.width = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style.height = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style[ Features.borderRadius ] = ( node._radius + node.getLineWidth() / 2 ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }
        }
      }

      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyRadius ) {
        this.matrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( -node._radius, -node._radius );
        this.matrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( this.matrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToClean();
    },

    onAttach: function( node ) {

    },

    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function( node ) {
      if ( !keepDOMCircleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }
    },

    setToClean: function() {
      this.setToCleanState();

      this.transformDirty = false;
    }
  } );

  /* jshint -W064 */
  Circle.CircleStatefulDrawable.mixin( CircleDOMDrawable );

  /* jshint -W064 */
  SelfDrawable.Poolable.mixin( CircleDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Circle.CircleSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function CircleSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: Circle.CircleStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'circle' );
      }
    },
    updateSVG: function( node, circle ) {
      if ( this.dirtyRadius ) {
        circle.setAttribute( 'r', node._radius );
      }

      this.updateFillStrokeStyle( circle );
    },
    usesPaint: true,
    keepElements: keepSVGCircleElements
  } );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Circle.CircleCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function CircleCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasCircle( wrapper, node ) {
      var context = wrapper.context;

      context.beginPath();
      context.arc( 0, 0, node._radius, 0, Math.PI * 2, false );
      context.closePath();

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
    },
    usesPaint: true,
    dirtyMethods: [ 'markDirtyRadius' ]
  } );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Circle.CircleWebGLDrawable = inherit( WebGLSelfDrawable, function CircleWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );

      //Small triangle strip that creates a square, which will be transformed into the right circle shape
      this.vertexCoordinates = this.vertexCoordinates || new Float32Array( [
        0, 0,
        1, 0,
        0, 1,
        1, 1
      ] );
    },

    initializeContext: function( gl ) {
      this.gl = gl;

      // cleanup old vertexBuffer, if applicable
      this.disposeWebGLBuffers();

      this.vertexBuffer = gl.createBuffer();
      this.initializePaintableState();
      this.updateCircle();
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateCircle: function() {
      var gl = this.gl;

      var circle = this.node;
      var rect = {
        _rectX: -circle.width / 2,
        _rectY: -circle.height / 2,
        _rectWidth: circle.width,
        _rectHeight: circle.height
      };

      this.vertexCoordinates[ 0 ] = rect._rectX;
      this.vertexCoordinates[ 1 ] = rect._rectY;

      this.vertexCoordinates[ 2 ] = rect._rectX + rect._rectWidth;
      this.vertexCoordinates[ 3 ] = rect._rectY;

      this.vertexCoordinates[ 4 ] = rect._rectX;
      this.vertexCoordinates[ 5 ] = rect._rectY + rect._rectHeight;

      this.vertexCoordinates[ 6 ] = rect._rectX + rect._rectWidth;
      this.vertexCoordinates[ 7 ] = rect._rectY + rect._rectHeight;

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      //TODO: Once we are lazily handling the full matrix, we may benefit from DYNAMIC draw here, and updating the vertices themselves
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexCoordinates, gl.STATIC_DRAW );

      // TODO: move to PaintableWebGLState???
      if ( this.dirtyFill ) {
        this.color = Color.toColor( this.node._fill );
        this.cleanPaintableState();
      }
    },

    render: function( shaderProgram ) {
      var gl = this.gl;

      // TODO: Handle rounded circles, please!
      // use the standard version if it's a rounded circle, since there is no WebGL-optimized version for that
      // TODO: how to handle fill/stroke delay optimizations here?
      if ( this.node._fill ) {
        //OHTWO TODO: optimize
        var viewMatrix = this.instance.relativeTransform.matrix.toAffineMatrix4();

        // combine image matrix (to scale aspect ratios), the trail's matrix, and the matrix to device coordinates
        gl.uniformMatrix4fv( shaderProgram.uniformLocations.uModelViewMatrix, false, viewMatrix.entries );

        //Indicate the branch of logic to use in the ubershader.  In this case, a texture should be used for the image
        gl.uniform1i( shaderProgram.uniformLocations.uFragmentType, WebGLBlock.fragmentTypeFill );
        gl.uniform4f( shaderProgram.uniformLocations.uColor, this.color.r / 255, this.color.g / 255, this.color.b / 255, this.color.a );

        gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
        gl.vertexAttribPointer( shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 0, 0 );


        phetAllocation && phetAllocation( 'drawArrays' );
        phetAllocation && phetAllocation( 'Circle.drawArrays' );
        gl.drawArrays( gl.TRIANGLE_STRIP, 0, 4 );
      }
    },

    shaderAttributes: [
      'aVertex'
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
      if ( this.gl ) {
        this.gl.deleteBuffer( this.vertexBuffer );
      }
    },

    markDirtyCircle: function() {
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
        this.updateCircle();
        this.dirty = false;
      }
    }
  } );

  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( Circle.CircleWebGLDrawable );

  // set up pooling
  SelfDrawable.Poolable.mixin( Circle.CircleWebGLDrawable );


  /*---------------------------------------------------------------------------*
   * Pixi Rendering
   *----------------------------------------------------------------------------*/

  Circle.CirclePixiDrawable = PixiSelfDrawable.createDrawable( {
    type: function CirclePixiDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: Circle.CircleStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      if ( !this.displayObject ) {
        this.displayObject = new PIXI.Graphics();
      }
    },
    updatePixi: function( node, circle ) {
      if ( this.dirtyRadius ) {
        var graphics = this.displayObject;
        this.displayObject.clear();
        if ( node.getFillColor() ) {
          graphics.beginFill( node.getFillColor().toNumber() );
        }
        if ( node.getStrokeColor() ) {
          graphics.lineStyle( 5, node.getStrokeColor().toNumber() );
        }
        graphics.drawRect( node.rectX, node.rectY, node.rectWidth, node.rectHeight );
        if ( node.getFillColor() ) {
          graphics.endFill();
        }
      }

      this.updateFillStrokeStyle( circle );
    },
    usesPaint: true,
    keepElements: keepSVGCircleElements
  } );

  return Circle;
} );
