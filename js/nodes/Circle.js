// Copyright 2002-2013, University of Colorado

/**
 * A circular node that inherits Path, and allows for optimized drawing,
 * and improved parameter handling.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Features = require( 'SCENERY/util/Features' );

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
      return bitmask;
    },

    getPathRendererBitmask: function() {
      return scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | ( Features.borderRadius ? scenery.bitmaskSupportsDOM : 0 );
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

    paintCanvas: function( wrapper ) {
      var context = wrapper.context;

      context.beginPath();
      context.arc( 0, 0, this._radius, 0, Math.PI * 2, false );
      context.closePath();

      if ( this._fill ) {
        this.beforeCanvasFill( wrapper ); // defined in Fillable
        context.fill();
        this.afterCanvasFill( wrapper ); // defined in Fillable
      }
      if ( this._stroke ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.stroke();
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },

    // create a circle instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'circle' );
    },

    // optimized for the circle element instead of path
    updateSVGFragment: function( circle ) {
      circle.setAttribute( 'r', this._radius );

      circle.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },

    /*---------------------------------------------------------------------------*
     * DOM support
     *----------------------------------------------------------------------------*/

    domUpdateTransformOnRepaint: true, // since we have to integrate the baseline offset into the CSS transform, signal to DOMLayer

    getDOMElement: function() {
      var fill = document.createElement( 'div' );
      var stroke = document.createElement( 'div' );
      fill.appendChild( stroke );
      fill.style.display = 'block';
      fill.style.position = 'absolute';
      fill.style.left = '0';
      fill.style.top = '0';
      stroke.style.display = 'block';
      stroke.style.position = 'absolute';
      stroke.style.left = '0';
      stroke.style.top = '0';
      return fill;
    },

    updateDOMElement: function( fill ) {
      fill.style.width = ( 2 * this._radius ) + 'px';
      fill.style.height = ( 2 * this._radius ) + 'px';
      fill.style[Features.borderRadius] = this._radius + 'px';
      fill.style.backgroundColor = this.getCSSFill();

      var stroke = fill.childNodes[0];
      if ( this.hasStroke() ) {
        stroke.style.width = ( 2 * this._radius - this.getLineWidth() ) + 'px';
        stroke.style.height = ( 2 * this._radius - this.getLineWidth() ) + 'px';
        stroke.style.left = ( -this.getLineWidth() / 2 ) + 'px';
        stroke.style.top = ( -this.getLineWidth() / 2 ) + 'px';
        stroke.style.borderStyle = 'solid';
        stroke.style.borderColor = this.getSimpleCSSFill();
        stroke.style.borderWidth = this.getLineWidth() + 'px';
        stroke.style[Features.borderRadius] = ( this._radius + this.getLineWidth() / 2 ) + 'px';
      }
      else {
        stroke.style.borderStyle = 'none';
      }
    },

    // override the transform since we need to customize it with a DOM offset
    updateCSSTransform: function( transform, element ) {
      // shift the text vertically, postmultiplied with the entire transform.
      var matrix = transform.getMatrix().timesMatrix( Matrix3.translation( -this._radius, -this._radius ) );
      scenery.Util.applyCSSTransform( matrix, element );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._radius + ', {' + propLines + '} )';
    },

    getRadius: function() {
      return this._radius;
    },

    setRadius: function( radius ) {
      if ( this._radius !== radius ) {
        this._radius = radius;
        this.invalidateCircle();
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

  return Circle;
} );
