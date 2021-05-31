// Copyright 2013-2021, University of Colorado Boulder

/**
 * A circular node that inherits Path, and allows for optimized drawing and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Shape from '../../../kite/js/Shape.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import CircleCanvasDrawable from '../display/drawables/CircleCanvasDrawable.js';
import CircleDOMDrawable from '../display/drawables/CircleDOMDrawable.js';
import CircleSVGDrawable from '../display/drawables/CircleSVGDrawable.js';
import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';
import Features from '../util/Features.js';
import Path from './Path.js';

const CIRCLE_OPTION_KEYS = [
  'radius' // {number} - see setRadius() for more documentation
];

class Circle extends Path {
  /**
   * @public
   * @mixes Paintable
   *
   * NOTE: There are two ways of invoking the constructor:
   * - new Circle( radius, { ... } )
   * - new Circle( { radius: radius, ... } )
   *
   * This allows the radius to be included in the parameter object for when that is convenient.
   *
   * @param {number} radius - The (non-negative) radius of the circle
   * @param {Object} [options] - Circle-specific options are documented in CIRCLE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  constructor( radius, options ) {
    super( null );

    // @private {number} - The radius of the circle
    this._radius = 0;

    // Handle new Circle( { radius: ... } )
    if ( typeof radius === 'object' ) {
      options = radius;
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );
    }
    // Handle new Circle( radius, { ... } )
    else {
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );
      options = extendDefined( {
        radius: radius
      }, options );
    }

    this.mutate( options );
  }


  /**
   * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
   * current stroke options.
   * @public (scenery-internal)
   * @override
   *
   * We can support the DOM renderer if there is a solid-styled stroke (which otherwise wouldn't be supported).
   *
   * @returns {number} - Renderer bitmask, see Renderer for details
   */
  getStrokeRendererBitmask() {
    let bitmask = super.getStrokeRendererBitmask();
    if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && this.getLineWidth() <= this.getRadius() ) {
      bitmask |= Renderer.bitmaskDOM;
    }
    return bitmask;
  }

  /**
   * Determines the allowed renderers that are allowed (or excluded) based on the current Path.
   * @public (scenery-internal)
   * @override
   *
   * @returns {number} - Renderer bitmask, see Renderer for details
   */
  getPathRendererBitmask() {
    // If we can use CSS borderRadius, we can support the DOM renderer.
    return Renderer.bitmaskCanvas | Renderer.bitmaskSVG | ( Features.borderRadius ? Renderer.bitmaskDOM : 0 );
  }

  /**
   * Notifies that the circle has changed (probably the radius), and invalidates path information and our cached
   * shape.
   * @private
   */
  invalidateCircle() {
    assert && assert( this._radius >= 0, 'A circle needs a non-negative radius' );

    // sets our 'cache' to null, so we don't always have to recompute our shape
    this._shape = null;

    // should invalidate the path and ensure a redraw
    this.invalidatePath();
  }

  /**
   * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
   * when one is needed, without having to do so beforehand.
   * @private
   *
   * @returns {Shape}
   */
  createCircleShape() {
    return Shape.circle( 0, 0, this._radius ).makeImmutable();
  }

  /**
   * Returns whether this Circle's selfBounds is intersected by the specified bounds.
   * @public
   *
   * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
   * @returns {boolean}
   */
  intersectsBoundsSelf( bounds ) {
    // TODO: handle intersection with somewhat-infinite bounds!
    let x = Math.abs( bounds.centerX );
    let y = Math.abs( bounds.centerY );
    const halfWidth = bounds.maxX - x;
    const halfHeight = bounds.maxY - y;

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
    CircleCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Circle.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {DOMSelfDrawable}
   */
  createDOMDrawable( renderer, instance ) {
    return CircleDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Circle.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {SVGSelfDrawable}
   */
  createSVGDrawable( renderer, instance ) {
    return CircleSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Circle.
   * @public (scenery-internal)
   * @override
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {CanvasSelfDrawable}
   */
  createCanvasDrawable( renderer, instance ) {
    return CircleCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Sets the radius of the circle.
   * @public
   *
   * @param {number} radius
   * @returns {Circle} - 'this' reference, for chaining
   */
  setRadius( radius ) {
    assert && assert( typeof radius === 'number', 'Circle.radius must be a number' );
    assert && assert( radius >= 0, 'A circle needs a non-negative radius' );
    assert && assert( isFinite( radius ), 'A circle needs a finite radius' );

    if ( this._radius !== radius ) {
      this._radius = radius;
      this.invalidateCircle();

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyRadius();
      }
    }
    return this;
  }

  set radius( value ) { this.setRadius( value ); }

  /**
   * Returns the radius of the circle.
   * @public
   *
   * @returns {number} - The radius of the circle
   */
  getRadius() {
    return this._radius;
  }

  get radius() { return this.getRadius(); }

  /**
   * Computes the bounds of the Circle, including any applied stroke. Overridden for efficiency.
   * @public
   * @override
   *
   * @returns {Bounds2}
   */
  computeShapeBounds() {
    let bounds = new Bounds2( -this._radius, -this._radius, this._radius, this._radius );
    if ( this._stroke ) {
      // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
      bounds = bounds.dilated( this.getLineWidth() / 2 );
    }
    return bounds;
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Circle's self content, or "outside".
   * @protected
   * @override
   *
   * Exists to optimize hit detection, as it's quick to compute for circles.
   *
   * @param {Vector2} point - Considered to be in the local coordinate frame
   * @returns {boolean}
   */
  containsPointSelf( point ) {
    const magSq = point.x * point.x + point.y * point.y;
    let result = true;
    let iRadius;
    if ( this._strokePickable ) {
      iRadius = this.getLineWidth() / 2;
      const outerRadius = this._radius + iRadius;
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
      const innerRadius = this._radius - iRadius;
      return result && magSq >= innerRadius * innerRadius;
    }
    else {
      return false; // neither stroke nor fill is pickable
    }
  }

  /**
   * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
   * parameters.
   * @public
   * @override
   *
   * @param {Shape|null} shape - Throws an error if it is not null.
   */
  setShape( shape ) {
    if ( shape !== null ) {
      throw new Error( 'Cannot set the shape of a Circle to something non-null' );
    }
    else {
      // probably called from the Path constructor
      this.invalidatePath();
    }
  }

  /**
   * Returns an immutable copy of this Path subtype's representation.
   * @public
   * @override
   *
   * NOTE: This is created lazily, so don't call it if you don't have to!
   *
   * @returns {Shape}
   */
  getShape() {
    if ( !this._shape ) {
      this._shape = this.createCircleShape();
    }
    return this._shape;
  }

  /**
   * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
   * @public
   * @override
   *
   * @returns {boolean}
   */
  hasShape() {
    // Always true for this Path subtype
    return true;
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
Circle.prototype._mutatorKeys = CIRCLE_OPTION_KEYS.concat( Path.prototype._mutatorKeys );

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * @public (scenery-internal)
 * @override
 */
Circle.prototype.drawableMarkFlags = Path.prototype.drawableMarkFlags.concat( [ 'radius' ] ).filter( flag => flag !== 'shape' );

scenery.register( 'Circle', Circle );

export default Circle;