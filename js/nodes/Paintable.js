// Copyright 2013-2016, University of Colorado Boulder

/**
 * Trait for nodes that support a standard fill and/or stroke (e.g. Text, Path and Path subtypes).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Color = require( 'SCENERY/util/Color' );
  var extend = require( 'PHET_CORE/extend' );
  var inheritance = require( 'PHET_CORE/inheritance' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  var Node = require( 'SCENERY/nodes/Node' );
  var PaintDef = require( 'SCENERY/util/PaintDef' );
  var platform = require( 'PHET_CORE/platform' );
  var Property = require( 'AXON/Property' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );

  var isSafari5 = platform.safari5;
  var isIE9 = platform.ie9;

  var PAINTABLE_OPTION_KEYS = [
    'fill', // Sets the fill of this node, see setFill() for documentation.
    'fillPickable', // Sets whether the filled area of the node will be treated as 'inside'. See setFillPickable()
    'stroke', // Sets the stroke of this node, see setStroke() for documentation.
    'strokePickable', // Sets whether the stroked area of the node will be treated as 'inside'. See setStrokePickable()
    'lineWidth', // Sets the width of the stroked area, see setLineWidth for documentation.
    'lineCap', // Sets the shape of the stroked area at the start/end of the path, see setLineCap() for documentation.
    'lineJoin', // Sets the shape of the stroked area at joints, see setLineJoin() for documentation.
    'miterLimit', // Sets when lineJoin will switch from miter to bevel, see setMiterLimit() for documentation.
    'lineDash', // Sets a line-dash pattern for the stroke, see setLineDash() for documentation
    'lineDashOffset', // Sets the offset of the line-dash from the start of the stroke, see setLineDashOffset()
    'cachedPaints' // Sets which paints should be cached, even if not displayed. See setCachedPaints()
  ];

  var DEFAULT_OPTIONS = {
    fill: null,
    fillPickable: true,
    stroke: null,
    strokePickable: false,

    // Not set initially, but they are the LineStyles defaults
    lineWidth: LineStyles.DEFAULT_OPTIONS.lineWidth,
    lineCap: LineStyles.DEFAULT_OPTIONS.lineCap,
    lineJoin: LineStyles.DEFAULT_OPTIONS.lineJoin,
    lineDashOffset: LineStyles.DEFAULT_OPTIONS.lineDashOffset,
    miterLimit: LineStyles.DEFAULT_OPTIONS.miterLimit
  };

  var Paintable = {
    /**
     * Applies the trait to a subtype of Node.
     * @public
     * @trait
     *
     * @param {constructor} type - A constructor that inherits from Node
     */
    mixInto: function( type ) {
      assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

      var proto = type.prototype;

      /**
       * These properties and methods are put directly on the prototype of things that have Paintable mixed in.
       */
      extend( proto, {
        /**
         * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
         *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
         *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
         * @public (scenery-internal)
         * @override
         */
        drawableMarkFlags: proto.drawableMarkFlags.concat( [ 'fill', 'stroke', 'lineWidth', 'lineOptions', 'cachedPaints' ] ),

        /**
         * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
         * order they will be evaluated in.
         * @protected
         *
         * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
         *       cases that may apply.
         */
        _mutatorKeys: PAINTABLE_OPTION_KEYS.concat( proto._mutatorKeys ),

        /**
         * This should be called in the constructor to initialize the paint-specific parts of the Node.
         * @protected
         */
        initializePaintable: function() {
          this._fill = DEFAULT_OPTIONS.fill;
          this._fillPickable = DEFAULT_OPTIONS.fillPickable;

          this._stroke = DEFAULT_OPTIONS.stroke;
          this._strokePickable = DEFAULT_OPTIONS.strokePickable;

          this._cachedPaints = [];
          this._lineDrawingStyles = new LineStyles();
        },

        /**
         * Sets the fill color for the node.
         * @public
         *
         * The fill determines the appearance of the interior part of a Path or Text.
         *
         * Please use null for indicating "no fill" (that is the default). Strings and Scenery Color objects can be
         * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
         * can also be provided.
         *
         * @param {PaintDef} fill
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setFill: function( fill ) {
          assert && assert( PaintDef.isPaintDef( fill ), 'Invalid fill type' );

          if ( assert && typeof fill === 'string' ) {
            Color.checkPaintString( fill );
          }

          // Instance equality used here since it would be more expensive to parse all CSS
          // colors and compare every time the fill changes. Right now, usually we don't have
          // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
          if ( this._fill !== fill ) {
            this._fill = fill;

            this.invalidateFill();
          }
          return this;
        },
        set fill( value ) { this.setFill( value ); },

        /**
         * Returns the fill (if any) for this Node.
         * @public
         *
         * @returns {PaintDef}
         */
        getFill: function() {
          return this._fill;
        },
        get fill() { return this.getFill(); },

        /**
         * Returns whether there is a fill applied to this Node.
         * @public
         *
         * @returns {boolean}
         */
        hasFill: function() {
          return this._fill !== null;
        },

        /**
         * Returns a property-unwrapped fill if applicable.
         * @public
         *
         * @returns {null|string|Color|LinearGradient|RadialGradient|Pattern}
         */
        getFillValue: function() {
          var fill = this.getFill();

          // Property lookup
          if ( fill instanceof Property ) {
            fill = fill.get();
          }

          return fill;
        },
        get fillValue() { return this.getFillValue(); },

        /**
         * Sets the stroke color for the node.
         * @public
         *
         * The stroke determines the appearance of the region along the boundary of the Path or Text. The shape of the
         * stroked area depends on the base shape (that of the Path or Text) and multiple parameters:
         * lineWidth/lineCap/lineJoin/miterLimit/lineDash/lineDashOffset. It will be drawn on top of any fill on the
         * same node.
         *
         * Please use null for indicating "no stroke" (that is the default). Strings and Scenery Color objects can be
         * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
         * can also be provided.
         *
         * @param {PaintDef} stroke
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setStroke: function( stroke ) {
          assert && assert( PaintDef.isPaintDef( stroke ), 'Invalid stroke type' );

          if ( assert && typeof stroke === 'string' ) {
            Color.checkPaintString( stroke );
          }

          // Instance equality used here since it would be more expensive to parse all CSS
          // colors and compare every time the fill changes. Right now, usually we don't have
          // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
          if ( this._stroke !== stroke ) {
            this._stroke = stroke;

            this.invalidateStroke();
          }
          return this;
        },
        set stroke( value ) { this.setStroke( value ); },

        /**
         * Returns the stroke (if any) for this Node.
         * @public
         *
         * @returns {PaintDef}
         */
        getStroke: function() {
          return this._stroke;
        },
        get stroke() { return this.getStroke(); },

        /**
         * Returns whether there is a stroke applied to this Node.
         * @public
         *
         * @returns {boolean}
         */
        hasStroke: function() {
          return this._stroke !== null;
        },

        /**
         * Returns whether there will appear to be a stroke for this Node. Properly handles the lineWidth:0 case.
         * @public
         *
         * @returns {boolean}
         */
        hasPaintableStroke: function() {
          // Should not be stroked if the lineWidth is 0, see https://github.com/phetsims/scenery/issues/658
          // and https://github.com/phetsims/scenery/issues/523
          return this.hasStroke() && this.getLineWidth() > 0;
        },

        /**
         * Returns a property-unwrapped stroke if applicable.
         * @public
         *
         * @returns {null|string|Color|LinearGradient|RadialGradient|Pattern}
         */
        getStrokeValue: function() {
          var stroke = this.getStroke();

          // Property lookup
          if ( stroke instanceof Property ) {
            stroke = stroke.get();
          }

          return stroke;
        },
        get strokeValue() { return this.getStrokeValue(); },

        /**
         * Sets whether the fill is marked as pickable.
         * @public
         *
         * @param {boolean} pickable
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setFillPickable: function( pickable ) {
          assert && assert( typeof pickable === 'boolean' );

          if ( this._fillPickable !== pickable ) {
            this._fillPickable = pickable;

            // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
            this.invalidateFill();
          }
          return this;
        },
        set fillPickable( value ) { this.setFillPickable( value ); },

        /**
         * Returns whether the fill is marked as pickable.
         * @public
         *
         * @returns {boolean}
         */
        isFillPickable: function() {
          return this._fillPickable;
        },
        get fillPickable() { return this.isFillPickable(); },

        /**
         * Sets whether the stroke is marked as pickable.
         * @public
         *
         * @param {boolean} pickable
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setStrokePickable: function( pickable ) {
          assert && assert( typeof pickable === 'boolean', 'strokePickable should be a boolean, not ' + pickable );

          if ( this._strokePickable !== pickable ) {
            this._strokePickable = pickable;

            // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
            this.invalidateStroke();
          }
          return this;
        },
        set strokePickable( value ) { this.setStrokePickable( value ); },

        /**
         * Returns whether the stroke is marked as pickable.
         * @public
         *
         * @returns {boolean}
         */
        isStrokePickable: function() {
          return this._strokePickable;
        },
        get strokePickable() { return this.isStrokePickable(); },

        /**
         * Sets the line width that will be applied to strokes on this Node.
         * @public
         *
         * @param {number} lineWidth
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineWidth: function( lineWidth ) {
          assert && assert( typeof lineWidth === 'number', 'lineWidth should be a number, not ' + lineWidth );
          assert && assert( lineWidth >= 0, 'lineWidth should be non-negative instead of ' + lineWidth );

          if ( this.getLineWidth() !== lineWidth ) {
            this._lineDrawingStyles.lineWidth = lineWidth;
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineWidth();
            }
          }
          return this;
        },
        set lineWidth( value ) { this.setLineWidth( value ); },

        /**
         * Returns the line width that would be applied to strokes.
         * @public
         *
         * @returns {number}
         */
        getLineWidth: function() {
          return this._lineDrawingStyles.lineWidth;
        },
        get lineWidth() { return this.getLineWidth(); },

        /**
         * Sets the line cap style. There are three options:
         * - 'butt' (the default) stops the line at the end point
         * - 'round' draws a semicircular arc around the end point
         * - 'square' draws a square outline around the end point (like butt, but extended by 1/2 line width out)
         * @public
         *
         * @param {string} lineCap
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineCap: function( lineCap ) {
          assert && assert( lineCap === 'butt' || lineCap === 'round' || lineCap === 'square',
            'lineCap should be one of "butt", "round" or "square", not ' + lineCap );

          if ( this._lineDrawingStyles.lineCap !== lineCap ) {
            this._lineDrawingStyles.lineCap = lineCap;
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineOptions();
            }
          }
          return this;
        },
        set lineCap( value ) { this.setLineCap( value ); },

        /**
         * Returns the line cap style (controls appearance at the start/end of paths)
         * @public
         *
         * @returns {string}
         */
        getLineCap: function() {
          return this._lineDrawingStyles.lineCap;
        },
        get lineCap() { return this.getLineCap(); },

        /**
         * Sets the line join style. There are three options:
         * - 'miter' (default) joins by extending the segments out in a line until they meet. For very sharp
         *           corners, they will be chopped off and will act like 'bevel', depending on what the miterLimit is.
         * - 'round' draws a circular arc to connect the two stroked areas.
         * - 'bevel' connects with a single line segment.
         * @public
         *
         * @param {string} lineJoin
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineJoin: function( lineJoin ) {
          assert && assert( lineJoin === 'miter' || lineJoin === 'round' || lineJoin === 'bevel',
            'lineJoin should be one of "miter", "round" or "bevel", not ' + lineJoin );

          if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
            this._lineDrawingStyles.lineJoin = lineJoin;
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineOptions();
            }
          }
          return this;
        },
        set lineJoin( value ) { this.setLineJoin( value ); },

        /**
         * Returns the current line join style (controls join appearance between drawn segments).
         * @public
         *
         * @returns {string}
         */
        getLineJoin: function() {
          return this._lineDrawingStyles.lineJoin;
        },
        get lineJoin() { return this.getLineJoin(); },

        /**
         * Sets the miterLimit value. This determines how sharp a corner with lineJoin: 'miter' will need to be before
         * it gets cut off to the 'bevel' behavior.
         * @public
         *
         * @param {number} miterLimit
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setMiterLimit: function( miterLimit ) {
          assert && assert( typeof miterLimit === 'number' && isFinite( miterLimit ), 'miterLimit should be a finite number' );

          if ( this._lineDrawingStyles.miterLimit !== miterLimit ) {
            this._lineDrawingStyles.miterLimit = miterLimit;
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineOptions();
            }
          }
          return this;
        },
        set miterLimit( value ) { this.setMiterLimit( value ); },

        /**
         * Returns the miterLimit value.
         * @public
         *
         * @returns {number}
         */
        getMiterLimit: function() {
          return this._lineDrawingStyles.miterLimit;
        },
        get miterLimit() { return this.getMiterLimit(); },

        /**
         * Sets the line dash pattern. Should be an array of numbers "on" and "off" alternating. An empty array
         * indicates no dashing.
         * @public
         *
         * @param {Array.<number>} lineDash
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineDash: function( lineDash ) {
          assert && assert( Array.isArray( lineDash ) && lineDash.every( function( n ) { return typeof n === 'number' && isFinite( n ) && n >= 0; } ),
            'lineDash should be an array of finite non-negative numbers' );

          if ( this._lineDrawingStyles.lineDash !== lineDash ) {
            this._lineDrawingStyles.lineDash = lineDash || [];
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineOptions();
            }
          }
          return this;
        },
        set lineDash( value ) { this.setLineDash( value ); },

        /**
         * Gets the line dash pattern. An empty array is the default, indicating no dashing.
         * @public
         *
         * @returns {Array.<number>}
         */
        getLineDash: function() {
          return this._lineDrawingStyles.lineDash;
        },
        get lineDash() { return this.getLineDash(); },

        /**
         * Returns whether the stroke will be dashed.
         * @public
         *
         * @returns {boolean}
         */
        hasLineDash: function() {
          return !!this._lineDrawingStyles.lineDash.length;
        },

        /**
         * Sets the offset of the line dash pattern from the start of the stroke. Defaults to 0.
         * @public
         *
         * @param {number} lineDashOffset
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineDashOffset: function( lineDashOffset ) {
          assert && assert( typeof lineDashOffset === 'number' && isFinite( lineDashOffset ),
            'lineDashOffset should be a number, not ' + lineDashOffset );

          if ( this._lineDrawingStyles.lineDashOffset !== lineDashOffset ) {
            this._lineDrawingStyles.lineDashOffset = lineDashOffset;
            this.invalidateStroke();

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyLineOptions();
            }
          }
          return this;
        },
        set lineDashOffset( value ) { this.setLineDashOffset( value ); },

        /**
         * Returns the offset of the line dash pattern from the start of the stroke.
         * @public
         *
         * @returns {number}
         */
        getLineDashOffset: function() {
          return this._lineDrawingStyles.lineDashOffset;
        },
        get lineDashOffset() { return this.getLineDashOffset(); },

        /**
         * Sets the LineStyles object (it determines stroke appearance). The passed-in object will be mutated as needed.
         * @public
         *
         * @param {LineStyles} lineStyles
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setLineStyles: function( lineStyles ) {
          assert && assert( lineStyles instanceof LineStyles );

          this._lineDrawingStyles = lineStyles;
          this.invalidateStroke();
          return this;
        },
        set lineStyles( value ) { this.setLineStyles( value ); },

        /**
         * Returns the composite {LineStyles} object, that determines stroke appearance.
         * @public
         *
         * @returns {LineStyles}
         */
        getLineStyles: function() {
          return this._lineDrawingStyles;
        },
        get lineStyles() { return this.getLineStyles(); },

        /**
         * Sets the cached paints to the input array (a defensive copy). Note that it also filters out fills that are
         * not considered paints (e.g. strings, Colors, etc.).
         * @public
         *
         * When this node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
         * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
         * SVG-side whenever the switch is made).
         *
         * Also note that duplicate paints are acceptable, and don't need to be filtered out before-hand.
         *
         * @param {Array.<string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern|null>} paints
         * @returns {Paintable} - Returns 'this' reference, for chaining
         */
        setCachedPaints: function( paints ) {
          this._cachedPaints = paints.filter( function( paint ) { return paint && paint.isPaint; } );

          var stateLen = this._drawables.length;
          for ( var i = 0; i < stateLen; i++ ) {
            this._drawables[ i ].markDirtyCachedPaints();
          }

          return this;
        },
        set cachedPaints( value ) { this.setCachedPaints( value ); },

        /**
         * Returns the cached paints.
         * @public
         *
         * @returns {Array.<string|Color|LinearGradient|RadialGradient|Pattern|null>}
         */
        getCachedPaints: function() {
          return this._cachedPaints;
        },
        get cachedPaints() { return this.getCachedPaints(); },

        /**
         * Adds a cached paint. Does nothing if paint is just a normal fill (string, Color), but for gradients and
         * patterns, it will be made faster to switch to.
         *
         * When this node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
         * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
         * SVG-side whenever the switch is made).
         *
         * Also note that duplicate paints are acceptible, and don't need to be filtered out before-hand.
         *
         * @param {string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern|null} paint
         */
        addCachedPaint: function( paint ) {
          if ( paint && paint.isPaint ) {
            this._cachedPaints.push( paint );

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyCachedPaints();
            }
          }
        },

        /**
         * Removes a cached paint. Does nothing if paint is just a normal fill (string, Color), but for gradients and
         * patterns it will remove any existing cached paint. If it was added more than once, it will need to be removed
         * more than once.
         *
         * When this node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
         * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
         * SVG-side whenever the switch is made).
         *
         * @param {string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern|null} paint
         */
        removeCachedPaint: function( paint ) {
          if ( paint && paint.isPaint ) {
            assert && assert( _.includes( this._cachedPaints, paint ) );

            arrayRemove( this._cachedPaints, paint );

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyCachedPaints();
            }
          }
        },

        /**
         * Applies the fill to a Canvas context wrapper, before filling.
         * @public (scenery-internal)
         *
         * @param {CanvasContextWrapper} wrapper
         */
        beforeCanvasFill: function( wrapper ) {
          var fillValue = this.getFillValue();

          wrapper.setFillStyle( fillValue );
          if ( fillValue.transformMatrix ) {
            wrapper.context.save();
            fillValue.transformMatrix.canvasAppendTransform( wrapper.context );
          }
        },

        /**
         * Unapplies the fill to a Canvas context wrapper, after filling.
         * @public (scenery-internal)
         *
         * @param {CanvasContextWrapper} wrapper
         */
        afterCanvasFill: function( wrapper ) {
          var fillValue = this.getFillValue();

          if ( fillValue.transformMatrix ) {
            wrapper.context.restore();
          }
        },

        /**
         * Applies the stroke to a Canvas context wrapper, before stroking.
         * @public (scenery-internal)
         *
         * @param {CanvasContextWrapper} wrapper
         */
        beforeCanvasStroke: function( wrapper ) {
          var strokeValue = this.getStrokeValue();

          // TODO: is there a better way of not calling so many things on each stroke?
          wrapper.setStrokeStyle( this._stroke );
          wrapper.setLineWidth( this.getLineWidth() );
          wrapper.setLineCap( this.getLineCap() );
          wrapper.setLineJoin( this.getLineJoin() );
          wrapper.setMiterLimit( this.getMiterLimit() );
          wrapper.setLineDash( this.getLineDash() );
          wrapper.setLineDashOffset( this.getLineDashOffset() );
          if ( strokeValue.transformMatrix ) {
            wrapper.context.save();
            strokeValue.transformMatrix.canvasAppendTransform( wrapper.context );
          }
        },

        /**
         * Unapplies the stroke to a Canvas context wrapper, after stroking.
         * @public (scenery-internal)
         *
         * @param {CanvasContextWrapper} wrapper
         */
        afterCanvasStroke: function( wrapper ) {
          var strokeValue = this.getStrokeValue();

          if ( strokeValue.transformMatrix ) {
            wrapper.context.restore();
          }
        },

        /**
         * If applicable, returns the CSS color for the fill.
         * @public
         *
         * @returns {string}
         */
        getCSSFill: function() {
          var fillValue = this.getFillValue();
          // if it's a Color object, get the corresponding CSS
          // 'transparent' will make us invisible if the fill is null
          return fillValue ? ( fillValue.toCSS ? fillValue.toCSS() : fillValue ) : 'transparent';
        },

        /**
         * If applicable, returns the CSS color for the stroke.
         * @public
         *
         * @returns {string}
         */
        getSimpleCSSStroke: function() {
          var strokeValue = this.getStrokeValue();
          // if it's a Color object, get the corresponding CSS
          // 'transparent' will make us invisible if the fill is null
          return strokeValue ? ( strokeValue.toCSS ? strokeValue.toCSS() : strokeValue ) : 'transparent';
        },

        /**
         * Returns the fill-specific property string for use with toString().
         * @protected (scenery-internal)
         * @override
         *
         * @param {string} spaces - Whitespace to add
         * @param {string} result
         * @returns {string}
         */
        appendFillablePropString: function( spaces, result ) {
          if ( this._fill ) {
            if ( result ) {
              result += ',\n';
            }
            if ( typeof this.getFillValue() === 'string' ) {
              result += spaces + 'fill: \'' + this.getFillValue() + '\'';
            }
            else {
              result += spaces + 'fill: ' + this.getFillValue().toString();
            }
          }

          return result;
        },

        /**
         * Returns the stroke-specific property string for use with toString().
         * @protected (scenery-internal)
         * @override
         *
         * @param {string} spaces - Whitespace to add
         * @param {string} result
         * @returns {string}
         */
        appendStrokablePropString: function( spaces, result ) {
          var self = this;

          function addProp( key, value, nowrap ) {
            if ( result ) {
              result += ',\n';
            }
            if ( !nowrap && typeof value === 'string' ) {
              result += spaces + key + ': \'' + value + '\'';
            }
            else {
              result += spaces + key + ': ' + value;
            }
          }

          if ( this._stroke ) {
            var defaultStyles = new LineStyles();
            if ( typeof this.getStrokeValue() === 'string' ) {
              addProp( 'stroke', this.getStrokeValue() );
            }
            else {
              addProp( 'stroke', this.getStrokeValue().toString(), true );
            }

            _.each( [ 'lineWidth', 'lineCap', 'miterLimit', 'lineJoin', 'lineDashOffset' ], function( prop ) {
              if ( self[ prop ] !== defaultStyles[ prop ] ) {
                addProp( prop, self[ prop ] );
              }
            } );

            if ( this.lineDash.length ) {
              addProp( 'lineDash', JSON.stringify( this.lineDash ), true );
            }
          }

          return result;
        },

        /**
         * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
         * current fill options.
         * @public (scenery-internal)
         *
         * This will be used for all types that directly mix in Paintable (i.e. Path and Text), but may be overridden
         * by subtypes.
         *
         * @returns {number} - Renderer bitmask, see Renderer for details
         */
        getFillRendererBitmask: function() {
          var bitmask = 0;

          // Safari 5 has buggy issues with SVG gradients
          if ( !( isSafari5 && this._fill && this._fill.isGradient ) ) {
            bitmask |= Renderer.bitmaskSVG;
          }

          // we always have Canvas support?
          bitmask |= Renderer.bitmaskCanvas;

          if ( !this.hasFill() ) {
            // if there is no fill, it is supported by DOM and WebGL
            bitmask |= Renderer.bitmaskDOM;
            bitmask |= Renderer.bitmaskWebGL;
          }
          else if ( this._fill.isPattern ) {
            // no pattern support for DOM or WebGL (for now!)
          }
          else if ( this._fill.isGradient ) {
            // no gradient support for DOM or WebGL (for now!)
          }
          else {
            // solid fills always supported for DOM and WebGL
            bitmask |= Renderer.bitmaskDOM;
            bitmask |= Renderer.bitmaskWebGL;
          }

          return bitmask;
        },

        /**
         * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
         * current stroke options.
         * @public (scenery-internal)
         *
         * This will be used for all types that directly mix in Paintable (i.e. Path and Text), but may be overridden
         * by subtypes.
         *
         * @returns {number} - Renderer bitmask, see Renderer for details
         */
        getStrokeRendererBitmask: function() {
          var bitmask = 0;

          // IE9 has bad dashed strokes, let's force a different renderer in case
          if ( !( isIE9 && this.hasStroke() && this.hasLineDash() ) ) {
            bitmask |= Renderer.bitmaskCanvas;
          }

          // always have SVG support (for now?)
          bitmask |= Renderer.bitmaskSVG;

          if ( !this.hasStroke() ) {
            // allow DOM support if there is no stroke (since the fill will determine what is available)
            bitmask |= Renderer.bitmaskDOM;
            bitmask |= Renderer.bitmaskWebGL;
          }

          return bitmask;
        }
      } );

      /**
       * Paintable's version of invalidateFill(), possibly combined with a client invalidateFill. Invalidates our
       * current fill, triggering recomputation of anything that depended on the old fill's value.
       * @protected
       */
      function invalidateFill() {
        this.invalidateSupportedRenderers();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyFill();
        }
      }

      // Patch in a sub-type call if it already exists on the prototype
      if ( proto.invalidateFill ) {
        var subtypeInvalidateFill = proto.invalidateFill;
        proto.invalidateFill = function() {
          subtypeInvalidateFill.call( this );
          invalidateFill.call( this );
        };
      }
      else {
        proto.invalidateFill = invalidateFill;
      }

      /**
       * Paintable's version of invalidateStroke(), possibly combined with a client invalidateStroke. Invalidates our
       * current stroke, triggering recomputation of anything that depended on the old stroke's value.
       * @protected
       */
      function invalidateStroke() {
        this.invalidateSupportedRenderers();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyStroke();
        }
      }

      // Patch in a sub-type call if it already exists on the prototype
      if ( proto.invalidateStroke ) {
        var subtypeInvalidateStroke = proto.invalidateStroke;
        proto.invalidateStroke = function() {
          subtypeInvalidateStroke.call( this );
          invalidateStroke.call( this );
        };
      }
      else {
        proto.invalidateStroke = invalidateStroke;
      }
    }
  };
  scenery.register( 'Paintable', Paintable );

  Paintable.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

  return Paintable;
} );
