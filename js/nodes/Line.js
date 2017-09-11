// Copyright 2013-2016, University of Colorado Boulder

/**
 * Displays a (stroked) line. Inherits Path, and allows for optimized drawing and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var inherit = require( 'PHET_CORE/inherit' );
  var KiteLine = require( 'KITE/segments/Line' ); // eslint-disable-line require-statement-match
  var LineCanvasDrawable = require( 'SCENERY/display/drawables/LineCanvasDrawable' );
  var LineSVGDrawable = require( 'SCENERY/display/drawables/LineSVGDrawable' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );
  var Vector2 = require( 'DOT/Vector2' );

  var LINE_OPTION_KEYS = [
    'p1', // {Vector2} - Start position
    'p2', // {Vector2} - End position
    'x1', // {number} - Start x position
    'y1', // {number} - Start y position
    'x2', // {number} - End x position
    'y2' // {number} - End y position
  ];

  /**
   * @public
   * @constructor
   * @extends Path
   *
   * Currently, all numerical parameters should be finite.
   * x1: {number} - x-position of the start
   * y1: {number} - y-position of the start
   * x2: {number} - x-position of the end
   * y2: {number} - y-position of the end
   * p1: {Vector2} - position of the start
   * p2: {Vector2} - position of the end
   *
   * Available constructors (with "..." denoting options parameters):
   * - new Line( x1, y1, x2, y2, { ... } )
   * - new Line( p1, p2, { ... } )
   * - A combination of options that sets all of the x's and y's, e.g.:
   *   - new Line( { p1: p1, p2: p2, ... } )
   *   - new Line( { p1: p1, x2: x2, y2: y2, ... } )
   *   - new Line( { x1: x1, y1: y1, x2: x2, y2: y2, ... } )
   *
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {Object} [options] - Line-specific options are documented in LINE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  function Line( x1, y1, x2, y2, options ) {
    // @private {number} - The x coordinate of the start point (point 1)
    this._x1 = 0;

    // @private {number} - The y coordinate of the start point (point 1)
    this._y1 = 0;

    // @private {number} - The x coordinate of the start point (point 2)
    this._x2 = 0;

    // @private {number} - The y coordinate of the start point (point 2)
    this._y2 = 0;

    // Remap constructor parameters to options
    if ( typeof x1 === 'object' ) {
      if ( x1 instanceof Vector2 ) {
        // assumes Line( Vector2, Vector2, options ), where x2 is our options
        assert && assert( y1 instanceof Vector2 );
        assert && assert( x2 === undefined || typeof x2 === 'object' );
        assert && assert( x2 === undefined || Object.getPrototypeOf( x2 ) === Object.prototype,
          'Extra prototype on Node options object is a code smell' );

        options = extendDefined( {
          // First Vector2 is under the x1 name
          x1: x1.x,
          y1: x1.y,
          // Second Vector2 is under the y1 name
          x2: y1.x,
          y2: y1.y
        }, x2 ); // Options object (if available) is under the x2 name
      }
      else {
        // assumes Line( { ... } ), init to zero for now
        assert && assert( y1 === undefined );

        // Options object is under the x1 name
        options = x1;

        assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
          'Extra prototype on Node options object is a code smell' );
      }
    }
    else {
      // new Line( x1, y1, x2, y2, [options] )
      assert && assert( typeof x1 === 'number' &&
                        typeof y1 === 'number' &&
                        typeof x2 === 'number' &&
                        typeof y2 === 'number' );
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );

      options = extendDefined( {
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2
      }, options );
    }

    Path.call( this, null, options );
  }

  scenery.register( 'Line', Line );

  inherit( Path, Line, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: LINE_OPTION_KEYS.concat( Path.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Path.prototype.drawableMarkFlags.concat( [ 'line', 'p1', 'p2', 'x1', 'x2', 'y1', 'y2' ] ).filter( function( flag ) {
      // We don't want the shape flag, as that won't be called for Path subtypes.
      return flag !== 'shape';
    } ),

    /**
     * Set all of the line's x and y values.
     * @public
     *
     * @param {number} x1 - the start x coordinate
     * @param {number} y1 - the start y coordinate
     * @param {number} x2 - the end x coordinate
     * @param {number} y2 - the end y coordinate
     * @returns {Line} - For chaining
     */
    setLine: function( x1, y1, x2, y2 ) {
      assert && assert( x1 !== undefined &&
                        y1 !== undefined &&
                        x2 !== undefined &&
                        y2 !== undefined, 'parameters need to be defined' );

      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyLine();
      }

      this.invalidateLine();

      return this;
    },

    /**
     * Set the line's first point's x and y values
     * @public
     *
     * Numeric parameters:
     * p1 {Vector2} - The first point
     * x1 {number} - The x coordinate of the first point
     * y1 {number} - THe y coordinate of the first point
     *
     * Available type signatures to call:
     * - setPoint1( x1, y1 )
     * - setPoint1( p1 )
     *
     * @param {number} x1 - the start x coordinate
     * @param {number} y1 - the start y coordinate
     * @returns {Line} - For chaining
     */
    setPoint1: function( x1, y1 ) {
      if ( typeof x1 === 'number' ) {
        // setPoint1( x1, y1 );
        assert && assert( x1 !== undefined && y1 !== undefined, 'parameters need to be defined' );
        this._x1 = x1;
        this._y1 = y1;
      }
      else {
        // setPoint1( Vector2 )
        assert && assert( x1.x !== undefined && x1.y !== undefined, 'parameters need to be defined' );
        this._x1 = x1.x;
        this._y1 = x1.y;
      }
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyP1();
      }
      this.invalidateLine();

      return this;
    },
    set p1( point ) { this.setPoint1( point ); },
    get p1() { return new Vector2( this._x1, this._y1 ); },

    /**
     * Set the line's second point's x and y values
     * @public
     *
     * Numeric parameters:
     * p2 {Vector2} - The second point
     * x2 {number} - The x coordinate of the second point
     * y2 {number} - THe y coordinate of the second point
     *
     * Available type signatures to call:
     * - setPoint2( x2, y2 )
     * - setPoint2( p2 )
     *
     * @param {number} x2 - the start x coordinate
     * @param {number} y2 - the start y coordinate
     * @returns {Line} - For chaining
     */
    setPoint2: function( x2, y2 ) {
      if ( typeof x2 === 'number' ) {
        // setPoint2( x2, y2 );
        assert && assert( x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );
        this._x2 = x2;
        this._y2 = y2;
      }
      else {
        // setPoint2( Vector2 )
        assert && assert( x2.x !== undefined && x2.y !== undefined, 'parameters need to be defined' );
        this._x2 = x2.x;
        this._y2 = x2.y;
      }
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyP2();
      }
      this.invalidateLine();

      return this;
    },
    set p2( point ) { this.setPoint2( point ); },
    get p2() { return new Vector2( this._x2, this._y2 ); },

    /**
     * Sets the x coordinate of the first point of the line.
     * @public
     *
     * @param {number} x1
     * @returns {Line} - For chaining.
     */
    setX1: function( x1 ) {
      if ( this._x1 !== x1 ) {
        this._x1 = x1;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyX1();
        }

        this.invalidateLine();
      }
      return this;
    },
    set x1( value ) { this.setX1( value ); },

    /**
     * Returns the x coordinate of the first point of the line.
     * @public
     *
     * @returns {number}
     */
    getX1: function() {
      return this._x1;
    },
    get x1() { return this.getX1(); },

    /**
     * Sets the y coordinate of the first point of the line.
     * @public
     *
     * @param {number} y1
     * @returns {Line} - For chaining.
     */
    setY1: function( y1 ) {
      if ( this._y1 !== y1 ) {
        this._y1 = y1;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyY1();
        }

        this.invalidateLine();
      }
      return this;
    },
    set y1( value ) { this.setY1( value ); },

    /**
     * Returns the y coordinate of the first point of the line.
     * @public
     *
     * @returns {number}
     */
    getY1: function() {
      return this._y1;
    },
    get y1() { return this.getY1(); },

    /**
     * Sets the x coordinate of the second point of the line.
     * @public
     *
     * @param {number} x2
     * @returns {Line} - For chaining.
     */
    setX2: function( x2 ) {
      if ( this._x2 !== x2 ) {
        this._x2 = x2;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyX2();
        }

        this.invalidateLine();
      }
      return this;
    },
    set x2( value ) { this.setX2( value ); },

    /**
     * Returns the x coordinate of the second point of the line.
     * @public
     *
     * @returns {number}
     */
    getX2: function() {
      return this._x2;
    },
    get x2() { return this.getX2(); },

    /**
     * Sets the y coordinate of the second point of the line.
     * @public
     *
     * @param {number} y2
     * @returns {Line} - For chaining.
     */
    setY2: function( y2 ) {
      if ( this._y2 !== y2 ) {
        this._y2 = y2;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyY2();
        }

        this.invalidateLine();
      }
      return this;
    },
    set y2( value ) { this.setY2( value ); },

    /**
     * Returns the y coordinate of the second point of the line.
     * @public
     *
     * @returns {number}
     */
    getY2: function() {
      return this._y2;
    },
    get y2() { return this.getY2(); },

    /**
     * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
     * when one is needed, without having to do so beforehand.
     * @private
     *
     * @returns {Shape}
     */
    createLineShape: function() {
      return Shape.lineSegment( this._x1, this._y1, this._x2, this._y2 ).makeImmutable();
    },

    /**
     * Notifies that the line has changed and invalidates path information and our cached shape.
     * @private
     */
    invalidateLine: function() {
      assert && assert( isFinite( this._x1 ), 'A rectangle needs to have a finite x1 (' + this._x1 + ')' );
      assert && assert( isFinite( this._y1 ), 'A rectangle needs to have a finite y1 (' + this._y1 + ')' );
      assert && assert( isFinite( this._x2 ), 'A rectangle needs to have a finite x2 (' + this._x2 + ')' );
      assert && assert( isFinite( this._y2 ), 'A rectangle needs to have a finite y2 (' + this._y2 + ')' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidatePath();
    },

    /**
     * Computes whether the provided point is "inside" (contained) in this Line's self content, or "outside".
     * @protected
     * @override
     *
     * Since an unstroked Line contains no area, we can quickly shortcut this operation.
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
    containsPointSelf: function( point ) {
      if ( this._strokePickable ) {
        return Path.prototype.containsPointSelf.call( this, point );
      }
      else {
        return false; // nothing is in a line! (although maybe we should handle edge points properly?)
      }
    },

    /**
     * Returns whether this Line's selfBounds is intersected by the specified bounds.
     * @public
     *
     * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
     * @returns {boolean}
     */
    intersectsBoundsSelf: function( bounds ) {
      // TODO: optimization
      return new KiteLine( this.p1, this.p2 ).intersectsBounds( bounds );
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
      LineCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
    },

    /**
     * Computes the bounds of the Line, including any applied stroke. Overridden for efficiency.
     * @public
     * @override
     *
     * @returns {Bounds2}
     */
    computeShapeBounds: function() {
      // optimized form for a single line segment (no joins, just two caps)
      if ( this._stroke ) {
        var lineCap = this.getLineCap();
        var halfLineWidth = this.getLineWidth() / 2;
        if ( lineCap === 'round' ) {
          // we can simply dilate by half the line width
          return new Bounds2(
            Math.min( this._x1, this._x2 ) - halfLineWidth, Math.min( this._y1, this._y2 ) - halfLineWidth,
            Math.max( this._x1, this._x2 ) + halfLineWidth, Math.max( this._y1, this._y2 ) + halfLineWidth );
        }
        else {
          // (dx,dy) is a vector p2-p1
          var dx = this._x2 - this._x1;
          var dy = this._y2 - this._y1;
          var magnitude = Math.sqrt( dx * dx + dy * dy );
          if ( magnitude === 0 ) {
            // if our line is a point, just dilate by halfLineWidth
            return new Bounds2( this._x1 - halfLineWidth, this._y1 - halfLineWidth, this._x2 + halfLineWidth, this._y2 + halfLineWidth );
          }
          // (sx,sy) is a vector with a magnitude of halfLineWidth pointed in the direction of (dx,dy)
          var sx = halfLineWidth * dx / magnitude;
          var sy = halfLineWidth * dy / magnitude;
          var bounds = Bounds2.NOTHING.copy();

          if ( lineCap === 'butt' ) {
            // four points just using the perpendicular stroked offsets (sy,-sx) and (-sy,sx)
            bounds.addCoordinates( this._x1 - sy, this._y1 + sx );
            bounds.addCoordinates( this._x1 + sy, this._y1 - sx );
            bounds.addCoordinates( this._x2 - sy, this._y2 + sx );
            bounds.addCoordinates( this._x2 + sy, this._y2 - sx );
          }
          else {
            assert && assert( lineCap === 'square' );

            // four points just using the perpendicular stroked offsets (sy,-sx) and (-sy,sx) and parallel stroked offsets
            bounds.addCoordinates( this._x1 - sx - sy, this._y1 - sy + sx );
            bounds.addCoordinates( this._x1 - sx + sy, this._y1 - sy - sx );
            bounds.addCoordinates( this._x2 + sx - sy, this._y2 + sy + sx );
            bounds.addCoordinates( this._x2 + sx + sy, this._y2 + sy - sx );
          }
          return bounds;
        }
      }
      else {
        // It might have a fill? Just include the fill bounds for now.
        var fillBounds = Bounds2.NOTHING.copy();
        fillBounds.addCoordinates( this._x1, this._y1 );
        fillBounds.addCoordinates( this._x2, this._y2 );
        return fillBounds;
      }
    },

    /**
     * Creates a SVG drawable for this Line.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return LineSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Line.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return LineCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
     * parameters.
     * @public
     * @override
     *
     * @param {Shape|null} Shape - Throws an error if it is not null.
     */
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Line to something non-null' );
      }
      else {
        // probably called from the Path constructor
        this.invalidatePath();
      }
    },

    /**
     * Returns an immutable copy of this Path subtype's representation.
     * @public
     * @override
     *
     * NOTE: This is created lazily, so don't call it if you don't have to!
     *
     * @returns {Shape}
     */
    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createLineShape();
      }
      return this._shape;
    },

    /**
     * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
     * @public
     * @override
     *
     * @returns {boolean}
     */
    hasShape: function() {
      return true;
    },

    /**
     * Returns available fill renderers.
     * @public (scenery-internal)
     * @override
     *
     * Since our line can't be filled, we support all fill renderers.
     *
     * @returns {number} - See Renderer for more information on the bitmasks
     */
    getFillRendererBitmask: function() {
      return Renderer.bitmaskCanvas | Renderer.bitmaskSVG | Renderer.bitmaskDOM | Renderer.bitmaskWebGL;
    }
  } );

  return Line;
} );
