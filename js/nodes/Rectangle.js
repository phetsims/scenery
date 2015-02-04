// Copyright 2002-2014, University of Colorado Boulder

/**
 * A rectangular node that inherits Path, and allows for optimized drawing,
 * and improved rectangle handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Dimension2 = require( 'DOT/Dimension2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Features = require( 'SCENERY/util/Features' );
  var Color = require( 'SCENERY/util/Color' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var PixiSelfDrawable = require( 'SCENERY/display/PixiSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var SquareUnstrokedRectangle = require( 'SCENERY/display/webgl/SquareUnstrokedRectangle' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMRectangleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGRectangleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  // scratch matrix used in DOM rendering
  var scratchMatrix = Matrix3.dirtyFromPool();

  /**
   * Currently, all numerical parameters should be finite.
   * x:         x-position of the upper-left corner (left bound)
   * y:         y-position of the upper-left corner (top bound)
   * width:     width of the rectangle to the right of the upper-left corner, required to be >= 0
   * height:    height of the rectangle below the upper-left corner, required to be >= 0
   * arcWidth:  positive width of the rounded corner, or 0 to indicate the corner should be sharp
   * arcHeight: positive height of the rounded corner, or 0 to indicate the corner should be sharp
   */
  scenery.Rectangle = function Rectangle( x, y, width, height, arcWidth, arcHeight, options ) {
    if ( typeof x === 'object' ) {
      if ( x instanceof Bounds2 ) {
        // allow new Rectangle( bounds2, { ... } ) or new Rectangle( bounds2, arcWidth, arcHeight, options )
        this._rectX = x.minX;
        this._rectY = x.minY;
        this._rectWidth = x.width;
        this._rectHeight = x.height;
        if ( arguments.length < 3 ) {
          // Rectangle( bounds2, { ... } )
          options = y;
          this._rectArcWidth = 0;
          this._rectArcHeight = 0;
        }
        else {
          // Rectangle( bounds2, arcWidth, arcHeight, { ... } )
          options = height;
          this._rectArcWidth = y;
          this._rectArcHeight = width;
        }
      }
      else {
        // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
        // the mutators will call invalidateRectangle() and properly set the shape
        options = x;
        this._rectX = options.rectX || 0;
        this._rectY = options.rectY || 0;
        this._rectWidth = options.rectWidth;
        this._rectHeight = options.rectHeight;
        this._rectArcWidth = options.rectArcWidth || 0;
        this._rectArcHeight = options.rectArcHeight || 0;
      }
    }
    else if ( arguments.length < 6 ) {
      // new Rectangle( x, y, width, height, [options] )
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = 0;
      this._rectArcHeight = 0;

      // ensure we have a parameter object
      options = arcWidth || {};

    }
    else {
      // normal case with args (including arcWidth / arcHeight)
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth;
      this._rectArcHeight = arcHeight;

      // ensure we have a parameter object
      options = options || {};

    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  };
  var Rectangle = scenery.Rectangle;

  inherit( Path, Rectangle, {

    getMaximumArcSize: function() {
      return Math.min( this._rectWidth / 2, this._rectHeight / 2 );
    },

    getStrokeRendererBitmask: function() {
      var bitmask = Path.prototype.getStrokeRendererBitmask.call( this );
      // DOM stroke handling doesn't YET support gradients, patterns, or dashes (with the current implementation, it shouldn't be too hard)
      if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && !this.hasLineDash() ) {
        // we can't support the bevel line-join with our current DOM rectangle display
        if ( this.getLineJoin() === 'miter' || ( this.getLineJoin() === 'round' && Features.borderRadius ) ) {
          bitmask |= scenery.bitmaskSupportsDOM;
        }
      }

      //TODO: Refine the rules for when WebGL can be used
      if ( !this.hasStroke() ) {
        bitmask |= scenery.bitmaskSupportsWebGL;
        bitmask |= scenery.bitmaskSupportsPixi;
      }
      return bitmask;
    },

    getPathRendererBitmask: function() {
      var bitmask = scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskBoundsValid;

      var maximumArcSize = this.getMaximumArcSize();

      // If the top/bottom or left/right strokes touch and overlap in the middle (small rectangle, big stroke), our DOM method won't work.
      // Additionally, if we're handling rounded rectangles or a stroke with lineJoin 'round', we'll need borderRadius
      // We also require for DOM that if it's a rounded rectangle, it's rounded with circular arcs (for now, could potentially do a transform trick!)
      if ( ( !this.hasStroke() || ( this.getLineWidth() <= this._rectHeight && this.getLineWidth() <= this._rectWidth ) ) &&
           ( !this.isRounded() || ( Features.borderRadius && this._rectArcWidth === this._rectArcHeight ) ) &&
           this._rectArcHeight <= maximumArcSize && this._rectArcWidth <= maximumArcSize ) {
        bitmask |= scenery.bitmaskSupportsDOM;
      }

      //only support WebGL if it's NOT rounded (for now) AND if it's either not stroked, or stroke has lineJoin !== round
      //TODO: Refine the rules for when WebGL can be used
      if ( !this.isRounded() && (!this.hasStroke() || this.getLineJoin() !== 'round') ) {
        bitmask |= scenery.bitmaskSupportsWebGL;
        bitmask |= scenery.bitmaskSupportsPixi;
      }

      return bitmask;
    },

    setRect: function( x, y, width, height, arcWidth, arcHeight ) {
      assert && assert( x !== undefined && y !== undefined && width !== undefined && height !== undefined, 'x/y/width/height need to be defined' );

      // for now, check whether this is needed
      // TODO: note that this could decrease performance? Remove if this is a bottleneck
      if ( this._rectX === x &&
           this._rectY === y &&
           this._rectWidth === width &&
           this._rectHeight === height &&
           this._rectArcWidth === arcWidth &&
           this._rectArcHeight === arcHeight ) {
        return;
      }

      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth || 0;
      this._rectArcHeight = arcHeight || 0;

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyRectangle();
      }
      this.invalidateRectangle();
    },

    // sets the Rectangle's x/y/width/height from the {Bounds2} bounds passed in.
    setRectBounds: function( bounds ) {
      assert && assert( bounds instanceof Bounds2 );

      this.setRect( bounds.x, bounds.y, bounds.width, bounds.height );
    },
    set rectBounds( value ) { this.setRectBounds( value ); },

    // gets a {Bounds2} from the Rectangle's x/y/width/height
    getRectBounds: function() {
      return Bounds2.rect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
    },
    get rectBounds() { return this.getRectBounds(); },

    // sets the Rectangle's width/height from the {Dimension2} size passed in.
    setRectSize: function( size ) {
      assert && assert( size instanceof Dimension2 );

      this.setRectWidth( size.width );
      this.setRectHeight( size.height );
    },
    set rectSize( value ) { this.setRectSize( value ); },

    // gets a {Dimension2} from the Rectangle's width/height
    getRectSize: function() {
      return new Dimension2( this._rectWidth, this._rectHeight );
    },
    get rectSize() { return this.getRectSize(); },

    // sets the width of the rectangle while keeping its right edge (x + width) in the same position
    setRectWidthFromRight: function( width ) {
      assert && assert( typeof width === 'number' );

      if ( this._rectWidth !== width ) {
        var right = this._rectX + this._rectWidth;
        this.setRectWidth( width );
        this.setRectX( right - width );
      }
    },
    set rectWidthFromRight( value ) { this.setRectWidthFromRight( value ); },
    get rectWidthFromRight() { return this.getRectWidth(); }, // because JSHint complains

    // sets the height of the rectangle while keeping its bottom edge (y + height) in the same position
    setRectHeightFromBottom: function( height ) {
      assert && assert( typeof height === 'number' );

      if ( this._rectHeight !== height ) {
        var bottom = this._rectY + this._rectHeight;
        this.setRectHeight( height );
        this.setRectY( bottom - height );
      }
    },
    set rectHeightFromBottom( value ) { this.setRectHeightFromBottom( value ); },
    get rectHeightFromBottom() { return this.getRectHeight(); }, // because JSHint complains

    isRounded: function() {
      return this._rectArcWidth !== 0 && this._rectArcHeight !== 0;
    },

    computeShapeBounds: function() {
      var bounds = new Bounds2( this._rectX, this._rectY, this._rectX + this._rectWidth, this._rectY + this._rectHeight );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
    },

    // @override, since Path's general function would slow us down (we're a rectangle, our local coordinate frame is perfect for this)
    getTransformedSelfBounds: function( matrix ) {
      return this._selfBounds.transformed( matrix );
    },

    createRectangleShape: function() {
      if ( this.isRounded() ) {
        // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
        var maximumArcSize = Math.min( this._rectWidth / 2, this._rectHeight / 2 );
        return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight,
          Math.min( maximumArcSize, this._rectArcWidth ), Math.min( maximumArcSize, this._rectArcHeight ) );
      }
      else {
        return Shape.rectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
      }
    },

    invalidateRectangle: function() {
      assert && assert( isFinite( this._rectX ), 'A rectangle needs to have a finite x (' + this._rectX + ')' );
      assert && assert( isFinite( this._rectY ), 'A rectangle needs to have a finite y (' + this._rectY + ')' );
      assert && assert( this._rectWidth >= 0 && isFinite( this._rectWidth ),
        'A rectangle needs to have a non-negative finite width (' + this._rectWidth + ')' );
      assert && assert( this._rectHeight >= 0 && isFinite( this._rectHeight ),
        'A rectangle needs to have a non-negative finite height (' + this._rectHeight + ')' );
      assert && assert( this._rectArcWidth >= 0 && isFinite( this._rectArcWidth ),
        'A rectangle needs to have a non-negative finite arcWidth (' + this._rectArcWidth + ')' );
      assert && assert( this._rectArcHeight >= 0 && isFinite( this._rectArcHeight ),
        'A rectangle needs to have a non-negative finite arcHeight (' + this._rectArcHeight + ')' );
      // assert && assert( !this.isRounded() || ( this._rectWidth >= this._rectArcWidth * 2 && this._rectHeight >= this._rectArcHeight * 2 ),
      //                                 'The rounded sections of the rectangle should not intersect (the length of the straight sections shouldn\'t be negative' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidateShape();

      // since we changed the rectangle arc width/height, it could make DOM work or not
      this.invalidateSupportedRenderers();
    },

    // accelerated hit detection for axis-aligned optionally-rounded rectangle
    // fast computation if it isn't rounded. if rounded, we check if a corner computation is needed (usually isn't), and only check that one needed corner
    containsPointSelf: function( point ) {
      var x = this._rectX;
      var y = this._rectY;
      var width = this._rectWidth;
      var height = this._rectHeight;
      var arcWidth = this._rectArcWidth;
      var arcHeight = this._rectArcHeight;
      var halfLine = this.getLineWidth() / 2;

      var result = true;
      if ( this._strokePickable ) {
        // test the outer boundary if we are stroke-pickable (if also fill-pickable, this is the only test we need)
        var rounded = this.isRounded();
        if ( !rounded && this.getLineJoin() === 'bevel' ) {
          // fall-back for bevel
          return Path.prototype.containsPointSelf.call( this, point );
        }
        var miter = this.getLineJoin() === 'miter' && !rounded;
        result = result && Rectangle.intersects( x - halfLine, y - halfLine,
          width + 2 * halfLine, height + 2 * halfLine,
          miter ? 0 : ( arcWidth + halfLine ), miter ? 0 : ( arcHeight + halfLine ),
          point );
      }

      if ( this._fillPickable ) {
        if ( this._strokePickable ) {
          return result;
        }
        else {
          return Rectangle.intersects( x, y, width, height, arcWidth, arcHeight, point );
        }
      }
      else if ( this._strokePickable ) {
        return result && !Rectangle.intersects( x + halfLine, y + halfLine,
            width - 2 * halfLine, height - 2 * halfLine,
            arcWidth - halfLine, arcHeight - halfLine,
            point );
      }
      else {
        return false; // either fill nor stroke is pickable
      }
    },

    intersectsBoundsSelf: function( bounds ) {
      return !this.computeShapeBounds().intersection( bounds ).isEmpty();
    },

    canvasPaintSelf: function( wrapper ) {
      Rectangle.RectangleCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createDOMDrawable: function( renderer, instance ) {
      return Rectangle.RectangleDOMDrawable.createFromPool( renderer, instance );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Rectangle.RectangleSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Rectangle.RectangleCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Rectangle.RectangleWebGLDrawable.createFromPool( renderer, instance );
    },

    createPixiDrawable: function( renderer, instance ) {
      return Rectangle.RectanglePixiDrawable.createFromPool( renderer, instance );
    },

    /*---------------------------------------------------------------------------*
     * Miscellaneous
     *----------------------------------------------------------------------------*/

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Rectangle( ' +
             this._rectX + ', ' + this._rectY + ', ' +
             this._rectWidth + ', ' + this._rectHeight + ', ' +
             this._rectArcWidth + ', ' + this._rectArcHeight +
             ', {' + propLines + '} )';
    },

    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Rectangle to something non-null' );
      }
      else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },

    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createRectangleShape();
      }
      return this._shape;
    },

    hasShape: function() {
      return true;
    }
  } );

  /*---------------------------------------------------------------------------*
   * Other Rectangle properties and ES5
   *----------------------------------------------------------------------------*/

  function addRectProp( capitalizedShort ) {
    var getName = 'getRect' + capitalizedShort;
    var setName = 'setRect' + capitalizedShort;
    var privateName = '_rect' + capitalizedShort;
    var dirtyMethodName = 'markDirty' + capitalizedShort;

    Rectangle.prototype[ getName ] = function() {
      return this[ privateName ];
    };

    Rectangle.prototype[ setName ] = function( value ) {
      if ( this[ privateName ] !== value ) {
        this[ privateName ] = value;
        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          var state = this._drawables[ i ];
          state[ dirtyMethodName ]();
          state.markPaintDirty();
        }
        this.invalidateRectangle();
      }
      return this;
    };

    Object.defineProperty( Rectangle.prototype, 'rect' + capitalizedShort, {
      set: Rectangle.prototype[ setName ],
      get: Rectangle.prototype[ getName ]
    } );
  }

  addRectProp( 'X' );
  addRectProp( 'Y' );
  addRectProp( 'Width' );
  addRectProp( 'Height' );
  addRectProp( 'ArcWidth' );
  addRectProp( 'ArcHeight' );

  // not adding mutators for now
  Rectangle.prototype._mutatorKeys = [ 'rectX', 'rectY', 'rectWidth', 'rectHeight', 'rectArcWidth', 'rectArcHeight' ].concat( Path.prototype._mutatorKeys );

  Rectangle.intersects = function( x, y, width, height, arcWidth, arcHeight, point ) {
    var result = point.x >= x &&
                 point.x <= x + width &&
                 point.y >= y &&
                 point.y <= y + height;

    if ( !result || arcWidth <= 0 || arcHeight <= 0 ) {
      return result;
    }

    // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
    var maximumArcSize = Math.min( width / 2, height / 2 );
    arcWidth = Math.min( maximumArcSize, arcWidth );
    arcHeight = Math.min( maximumArcSize, arcHeight );

    // we are rounded and inside the logical rectangle (if it didn't have rounded corners)

    // closest corner arc's center (we assume the rounded rectangle's arcs are 90 degrees fully, and don't intersect)
    var closestCornerX, closestCornerY, guaranteedInside = false;

    // if we are to the inside of the closest corner arc's center, we are guaranteed to be in the rounded rectangle (guaranteedInside)
    if ( point.x < x + width / 2 ) {
      closestCornerX = x + arcWidth;
      guaranteedInside = guaranteedInside || point.x >= closestCornerX;
    }
    else {
      closestCornerX = x + width - arcWidth;
      guaranteedInside = guaranteedInside || point.x <= closestCornerX;
    }
    if ( guaranteedInside ) { return true; }

    if ( point.y < y + height / 2 ) {
      closestCornerY = y + arcHeight;
      guaranteedInside = guaranteedInside || point.y >= closestCornerY;
    }
    else {
      closestCornerY = y + height - arcHeight;
      guaranteedInside = guaranteedInside || point.y <= closestCornerY;
    }
    if ( guaranteedInside ) { return true; }

    // we are now in the rectangular region between the logical corner and the center of the closest corner's arc.

    // offset from the closest corner's arc center
    var offsetX = point.x - closestCornerX;
    var offsetY = point.y - closestCornerY;

    // normalize the coordinates so now we are dealing with a unit circle
    // (technically arc, but we are guaranteed to be in the area covered by the arc, so we just consider the circle)
    // NOTE: we are rounded, so both arcWidth and arcHeight are non-zero (this is well defined)
    offsetX /= arcWidth;
    offsetY /= arcHeight;

    offsetX *= offsetX;
    offsetY *= offsetY;
    return offsetX + offsetY <= 1; // return whether we are in the rounded corner. see the formula for an ellipse
  };

  Rectangle.rect = function( x, y, width, height, options ) {
    return new Rectangle( x, y, width, height, 0, 0, options );
  };

  Rectangle.roundedRect = function( x, y, width, height, arcWidth, arcHeight, options ) {
    return new Rectangle( x, y, width, height, arcWidth, arcHeight, options );
  };

  Rectangle.bounds = function( bounds, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, 0, 0, options );
  };

  Rectangle.roundedBounds = function( bounds, arcWidth, arcHeight, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, arcWidth, arcHeight, options );
  };

  /*---------------------------------------------------------------------------*
   * Rendering state mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  Rectangle.RectangleStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyX = true;
        this.dirtyY = true;
        this.dirtyWidth = true;
        this.dirtyHeight = true;
        this.dirtyArcWidth = true;
        this.dirtyArcHeight = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState();

        return this; // allow for chaining
      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyRectangle = function() {
        // TODO: consider bitmask instead?
        this.dirtyX = true;
        this.dirtyY = true;
        this.dirtyWidth = true;
        this.dirtyHeight = true;
        this.dirtyArcWidth = true;
        this.dirtyArcHeight = true;
        this.markPaintDirty();
      };

      proto.markDirtyX = function() {
        this.dirtyX = true;
        this.markPaintDirty();
      };
      proto.markDirtyY = function() {
        this.dirtyY = true;
        this.markPaintDirty();
      };
      proto.markDirtyWidth = function() {
        this.dirtyWidth = true;
        this.markPaintDirty();
      };
      proto.markDirtyHeight = function() {
        this.dirtyHeight = true;
        this.markPaintDirty();
      };
      proto.markDirtyArcWidth = function() {
        this.dirtyArcWidth = true;
        this.markPaintDirty();
      };
      proto.markDirtyArcHeight = function() {
        this.dirtyArcHeight = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyX = false;
        this.dirtyY = false;
        this.dirtyWidth = false;
        this.dirtyHeight = false;
        this.dirtyArcWidth = false;
        this.dirtyArcHeight = false;

        this.cleanPaintableState();
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  var RectangleDOMDrawable = Rectangle.RectangleDOMDrawable = inherit( DOMSelfDrawable, function RectangleDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState();

      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.fillElement || !this.strokeElement ) {
        var fillElement = this.fillElement = document.createElement( 'div' );
        fillElement.style.display = 'block';
        fillElement.style.position = 'absolute';
        fillElement.style.left = '0';
        fillElement.style.top = '0';
        fillElement.style.pointerEvents = 'none';

        var strokeElement = this.strokeElement = document.createElement( 'div' );
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
        var borderRadius = Math.min( node._rectArcWidth, node._rectArcHeight );
        var borderRadiusDirty = this.dirtyArcWidth || this.dirtyArcHeight;

        if ( this.dirtyWidth ) {
          fillElement.style.width = node._rectWidth + 'px';
        }
        if ( this.dirtyHeight ) {
          fillElement.style.height = node._rectHeight + 'px';
        }
        if ( borderRadiusDirty ) {
          fillElement.style[ Features.borderRadius ] = borderRadius + 'px'; // if one is zero, we are not rounded, so we do the min here
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
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only rectangles)
          var hadNoStrokeBefore = this.lastStroke === null;

          if ( hadNoStrokeBefore || this.dirtyWidth || this.dirtyLineWidth ) {
            strokeElement.style.width = ( node._rectWidth - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyHeight || this.dirtyLineWidth ) {
            strokeElement.style.height = ( node._rectHeight - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }

          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }

          if ( hadNoStrokeBefore || borderRadiusDirty || this.dirtyLineWidth || this.dirtyLineOptions ) {
            strokeElement.style[ Features.borderRadius ] = ( node.isRounded() || node.getLineJoin() === 'round' ) ? ( borderRadius + node.getLineWidth() / 2 ) + 'px' : '0';
          }
        }
      }

      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyX || this.dirtyY ) {
        scratchMatrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( node._rectX, node._rectY );
        scratchMatrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( scratchMatrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToClean();
    },

    onAttach: function( node ) {

    },

    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function( node ) {
      if ( !keepDOMRectangleElements ) {
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

  Rectangle.RectangleStatefulDrawable.mixin( RectangleDOMDrawable );

  SelfDrawable.Poolable.mixin( RectangleDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG rendering
   *----------------------------------------------------------------------------*/

  Rectangle.RectangleSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function RectangleSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: Rectangle.RectangleStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      this.lastArcW = -1; // invalid on purpose
      this.lastArcH = -1; // invalid on purpose

      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'rect' );
      }
    },
    updateSVG: function( node, rect ) {
      if ( this.dirtyX ) {
        rect.setAttribute( 'x', node._rectX );
      }
      if ( this.dirtyY ) {
        rect.setAttribute( 'y', node._rectY );
      }
      if ( this.dirtyWidth ) {
        rect.setAttribute( 'width', node._rectWidth );
      }
      if ( this.dirtyHeight ) {
        rect.setAttribute( 'height', node._rectHeight );
      }
      if ( this.dirtyArcWidth || this.dirtyArcHeight || this.dirtyWidth || this.dirtyHeight ) {
        var arcw = 0;
        var arch = 0;

        // workaround for various browsers if rx=20, ry=0 (behavior is inconsistent, either identical to rx=20,ry=20, rx=0,ry=0. We'll treat it as rx=0,ry=0)
        // see https://github.com/phetsims/scenery/issues/183
        if ( node.isRounded() ) {
          var maximumArcSize = node.getMaximumArcSize();
          arcw = Math.min( node._rectArcWidth, maximumArcSize );
          arch = Math.min( node._rectArcHeight, maximumArcSize );
        }
        if ( arcw !== this.lastArcW ) {
          this.lastArcW = arcw;
          rect.setAttribute( 'rx', arcw );
        }
        if ( arch !== this.lastArcH ) {
          this.lastArcH = arch;
          rect.setAttribute( 'ry', arch );
        }
      }

      this.updateFillStrokeStyle( rect );
    },
    usesPaint: true,
    keepElements: keepSVGRectangleElements
  } );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Rectangle.RectangleCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function RectangleCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasRectangle( wrapper, node ) {
      var context = wrapper.context;

      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( node.isRounded() ) {
        context.beginPath();
        var maximumArcSize = node.getMaximumArcSize();
        var arcw = Math.min( node._rectArcWidth, maximumArcSize );
        var arch = Math.min( node._rectArcHeight, maximumArcSize );
        var lowX = node._rectX + arcw;
        var highX = node._rectX + node._rectWidth - arcw;
        var lowY = node._rectY + arch;
        var highY = node._rectY + node._rectHeight - arch;
        if ( arcw === arch ) {
          // we can use circular arcs, which have well defined stroked offsets
          context.arc( highX, lowY, arcw, -Math.PI / 2, 0, false );
          context.arc( highX, highY, arcw, 0, Math.PI / 2, false );
          context.arc( lowX, highY, arcw, Math.PI / 2, Math.PI, false );
          context.arc( lowX, lowY, arcw, Math.PI, Math.PI * 3 / 2, false );
        }
        else {
          // we have to resort to elliptical arcs
          context.ellipse( highX, lowY, arcw, arch, 0, -Math.PI / 2, 0, false );
          context.ellipse( highX, highY, arcw, arch, 0, 0, Math.PI / 2, false );
          context.ellipse( lowX, highY, arcw, arch, 0, Math.PI / 2, Math.PI, false );
          context.ellipse( lowX, lowY, arcw, arch, 0, Math.PI, Math.PI * 3 / 2, false );
        }
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
      }
      else {
        // TODO: how to handle fill/stroke delay optimizations here?
        if ( node._fill ) {
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fillRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }
        if ( node._stroke ) {
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.strokeRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
      }
    },
    usesPaint: true,
    dirtyMethods: [ 'markDirtyRectangle' ]
  } );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Rectangle.RectangleWebGLDrawable = inherit( WebGLSelfDrawable, function RectangleWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    initializeContext: function( webglBlock ) {
      this.webglBlock = webglBlock;
      this.rectangleHandle = new SquareUnstrokedRectangle( webglBlock.webglRenderer.colorTriangleRenderer, this.node, 0.5 );

      // cleanup old vertexBuffer, if applicable
      this.disposeWebGLBuffers();

      this.initializePaintableState();
      this.updateRectangle();

      //TODO: Update the state in the buffer arrays
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateRectangle: function() {

      // TODO: a way to update the ColorTriangleBufferData.

      // TODO: move to PaintableWebGLState???
      if ( this.dirtyFill ) {
        this.color = Color.toColor( this.node._fill );
        this.cleanPaintableState();
      }
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

  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( Rectangle.RectangleWebGLDrawable );

  // set up pooling
  SelfDrawable.Poolable.mixin( Rectangle.RectangleWebGLDrawable );

  /*---------------------------------------------------------------------------*
   * Pixi rendering
   *----------------------------------------------------------------------------*/

  Rectangle.RectanglePixiDrawable = PixiSelfDrawable.createDrawable( {
    type: function RectanglePixiDrawable( renderer, instance ) {
      this.initialize( renderer, instance );
    },
    stateType: Rectangle.RectangleStatefulDrawable.mixin,
    initialize: function( renderer, instance ) {
      this.lastArcW = -1; // invalid on purpose
      this.lastArcH = -1; // invalid on purpose

      if ( !this.displayObject ) {
        this.displayObject = new PIXI.Graphics();
      }
    },
    updatePixi: function( node, rect ) {
      if ( this.dirtyX || this.dirtyY || this.dirtyWidth || this.dirtyHeight ||
           this.dirtyArcWidth || this.dirtyArcHeight || this.dirtyWidth || this.dirtyHeight ) {
        var graphics = this.displayObject;
        this.displayObject.clear();
        graphics.beginFill( node.getFillColor().toNumber() );
        graphics.drawRect( node.rectX, node.rectY + 660, node.rectWidth, node.rectHeight );
        graphics.endFill();
      }
      this.updateFillStrokeStyle( rect );
    },
    usesPaint: true,
    keepElements: keepSVGRectangleElements
  } );

  return Rectangle;
} );
