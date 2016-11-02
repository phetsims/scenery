// Copyright 2013-2015, University of Colorado Boulder

/**
 * Mix-in for nodes that support a standard fill and/or stroke.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var Color = require( 'SCENERY/util/Color' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  var Renderer = require( 'SCENERY/display/Renderer' );

  var inherit = require( 'PHET_CORE/inherit' );
  var extend = require( 'PHET_CORE/extend' );
  var platform = require( 'PHET_CORE/platform' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );

  var Property = require( 'AXON/Property' );

  var isSafari5 = platform.safari5;
  var isIE9 = platform.ie9;

  /*
   * Applies the mix-in to a subtype of Node.
   * @public
   *
   * @param {constructor} type - A constructor that inherits from Node
   */
  var Paintable = {
    mixin: function( type ) {
      var proto = type.prototype;

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
         * This should be called in the constructor to initialize the paint-specific parts of the Node.
         * @protected
         */
        initializePaintable: function() {
          this._fill = null;
          this._fillPickable = true;

          this._stroke = null;
          this._strokePickable = false;

          this._cachedPaints = [];
          this._lineDrawingStyles = new LineStyles();
        },

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
         * Returns whether there is a stroke applied to this Node.
         * @public
         *
         * @returns {boolean}
         */
        hasStroke: function() {
          return this._stroke !== null;
        },

        /**
         * Returns the fill (if any) for this Node.
         * @public
         *
         * @returns {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
         */
        getFill: function() {
          return this._fill;
        },
        get fill() { return this.getFill(); },

        /**
         * Sets the fill color for the node.
         * @public
         *
         * Please use null for indicating "no fill" (that is the default). Strings and Scenery Color objects can be
         * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
         * can also be provided.
         *
         * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} fill
         */
        setFill: function( fill ) {
          assert && assert( fill === null ||
                            typeof fill === 'string' ||
                            fill instanceof Color ||
                            fill.isPaint ||
                            ( ( fill instanceof Property ) && (
                              typeof fill.value === 'string' ||
                              fill.value instanceof Color
                            ) ),
            'Invalid fill type' );

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
         * Returns the stroke (if any) for this Node.
         * @public
         *
         * @returns {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
         */
        getStroke: function() {
          return this._stroke;
        },
        get stroke() { return this.getStroke(); },

        /**
         * Sets the stroke color for the node.
         * @public
         *
         * Please use null for indicating "no stroke" (that is the default). Strings and Scenery Color objects can be
         * provided for a single-color flat appearance, and can be wrapped with an Axon Property. Gradients and patterns
         * can also be provided.
         *
         * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} stroke
         */
        setStroke: function( stroke ) {
          assert && assert( stroke === null ||
                            typeof stroke === 'string' ||
                            stroke instanceof Color ||
                            stroke.isPaint ||
                            ( ( stroke instanceof Property ) && (
                              typeof stroke.value === 'string' ||
                              stroke.value instanceof Color
                            ) ),
            'Invalid stroke type' );

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
         * Sets whether the fill is marked as pickable.
         * @public
         *
         * @param {boolean} pickable
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
         * Sets whether the stroke is marked as pickable.
         * @public
         *
         * @param {boolean} pickable
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
         * Sets the line width that will be applied to strokes on this Node.
         * @public
         *
         * @param {number} lineWidth
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
         * Sets the line cap style. There are three options:
         * - 'butt' (the default) stops the line at the end point
         * - 'round' draws a semicircular arc around the end point
         * - 'square' draws a square outline around the end point (like butt, but extended by 1/2 line width out)
         * @public
         *
         * @param {string} lineCap
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
         * Sets the line join style. There are three options:
         * - 'miter' (default) joins by extending the segments out in a line until they meet. For very sharp
         *           corners, they will be chopped off and will act like 'bevel', depending on what the miterLimit is.
         * - 'round' draws a circular arc to connect the two stroked areas.
         * - 'bevel' connects with a single line segment.
         * @public
         *
         * @param {string} lineJoin
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
         * Sets the miterLimit value. This determines how sharp a corner with lineJoin: 'miter' will need to be before
         * it gets cut off to the 'bevel' behavior.
         * @public
         *
         * @param {number} miterLimit
         */
        setMiterLimit: function( miterLimit ) {
          assert && assert( typeof miterLimit === 'number' );

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
         * Returns whether the stroke will be dashed.
         * @public
         *
         * @returns {boolean}
         */
        hasLineDash: function() {
          return !!this._lineDrawingStyles.lineDash.length;
        },

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
         * Sets the line dash pattern. Should be an array of numbers "on" and "off" alternating. An empty array
         * indicates no dashing.
         * @public
         *
         * @param {Array.<number>} lineDash
         */
        setLineDash: function( lineDash ) {
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
         * Sets the offset of the line dash pattern from the start of the stroke. Defaults to 0.
         * @public
         *
         * @param {number} lineDashOffset
         */
        setLineDashOffset: function( lineDashOffset ) {
          assert && assert( typeof lineDashOffset === 'number', 'lineDashOffset should be a number, not ' + lineDashOffset );

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
         * Sets the LineStyles object (it determines stroke appearance). The passed-in object will be mutated as needed.
         * @public
         *
         * @param {LineStyles} lineStyles
         */
        setLineStyles: function( lineStyles ) {
          this._lineDrawingStyles = lineStyles;
          this.invalidateStroke();
          return this;
        },
        set lineStyles( value ) { this.setLineStyles( value ); },

        /**
         * Returns the cached paints.
         * @public
         *
         * @returns {Array.<string|Color|LinearGradient|RadialGradient|Pattern|null}
         */
        getCachedPaints: function() {
          return this._cachedPaints;
        },
        get cachedPaints() { return this.getCachedPaints(); },

        /**
         * Sets the cached paints to the input array (a defensive copy). Note that it also filters out fills that are
         * not considered paints (e.g. strings, Colors, etc.).
         * @public
         *
         * When this node is displayed in SVG, it will force the presence of the cached paint to be stored in the SVG's
         * <defs> element, so that we can switch quickly to use the given paint (instead of having to create it on the
         * SVG-side whenever the switch is made).
         *
         * Also note that duplicate paints are acceptible, and don't need to be filtered out before-hand.
         *
         * @param {Array.<string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern|null>} paints
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
            assert && assert( _.contains( this._cachedPaints, paint ) );

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

      // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
      // TODO: docs
      proto._mutatorKeys = [
        'fill', 'fillPickable', 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'miterLimit', 'lineDash',
        'lineDashOffset', 'strokePickable', 'cachedPaints'
      ].concat( proto._mutatorKeys );

      // Paintable's version of invalidateFill()
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

      // Paintable's version of invalidateStroke()
      function invalidateStroke() {
        this.invalidateSupportedRenderers();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyStroke();
        }
      }

      // TODO: document!
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

  /**
   * An observer for a fill or stroke, that will be able to trigger notifications when it changes.
   */
  Paintable.PaintObserver = function PaintObserver( type, changeCallback ) {
    assert && assert( type === 'fill' || type === 'stroke' );
    this.type = type;
    this.name = '_' + type;
    this.changeCallback = changeCallback;
    this.primary = null;
    this.secondary = null;
    this.updateListener = this.update.bind( this );
  };
  inherit( Object, Paintable.PaintObserver, {
    initialize: function( node ) {
      assert && assert( node !== null );
      this.node = node;

      this.update();
    },

    update: function() {
      var primary = this.node[ this.name ];
      if ( primary !== this.primary ) {
        this.detachPrimary( this.primary );
        this.attachPrimary( primary );
        this.changeCallback();
      }
      else if ( primary instanceof Property ) {
        var secondary = primary.get();
        if ( secondary !== this.secondary ) {
          this.detachSecondary( this.secondary );
          this.attachSecondary( secondary );
          this.changeCallback();
        }
      }
    },

    attachPrimary: function( paint ) {
      this.primary = paint;
      if ( paint instanceof Property ) {
        paint.lazyLink( this.updateListener );
        this.attachSecondary( paint.get() );
      }
      else if ( paint instanceof Color ) {
        paint.addChangeListener( this.changeCallback );
      }
    },

    detachPrimary: function( paint ) {
      if ( paint instanceof Property ) {
        paint.unlink( this.updateListener );
        this.detachSecondary( paint.get() );
        this.secondary = null;
      }
      else if ( paint instanceof Color ) {
        paint.removeChangeListener( this.changeCallback );
      }
      this.primary = null;
    },

    attachSecondary: function( paint ) {
      this.secondary = paint;
      if ( paint instanceof Color ) {
        paint.addChangeListener( this.changeCallback );
      }
    },

    detachSecondary: function( paint ) {
      if ( paint instanceof Color ) {
        paint.removeChangeListener( this.changeCallback );
      }
      this.secondary = null;
    },

    clean: function() {
      this.detachPrimary( this.primary );
      this.node = null;
    }
  } );

  // mix-in base for DOM and SVG drawables
  // NOTE: requires state.node to be defined
  // TODO: doc!
  Paintable.PaintableStatefulDrawable = {
    mixin: function PaintableStatefulDrawable( drawableType ) {
      var proto = drawableType.prototype;

      proto.initializePaintableState = function( renderer, instance ) {
        this.lastFill = undefined;
        this.dirtyFill = true;

        this.lastStroke = undefined;
        this.dirtyStroke = true;
        this.dirtyLineWidth = true;
        this.dirtyLineOptions = true; // e.g. cap, join, dash, dashoffset, miterlimit
        this.dirtyCachedPaints = true;
        this.lastCachedPaints = [];

        this.fillCallback = this.fillCallback || this.markDirtyFill.bind( this );
        this.strokeCallback = this.strokeCallback || this.markDirtyStroke.bind( this );
        this.fillObserver = this.fillObserver || new Paintable.PaintObserver( 'fill', this.fillCallback );
        this.strokeObserver = this.strokeObserver || new Paintable.PaintObserver( 'stroke', this.strokeCallback );

        this.fillObserver.initialize( instance.node );
        this.strokeObserver.initialize( instance.node );

        return this;
      };

      proto.cleanPaintableState = function() {
        this.dirtyFill = false;
        this.lastFill = this.node.getFill();

        this.dirtyStroke = false;
        this.dirtyLineWidth = false;
        this.dirtyLineOptions = false;
        this.dirtyCachedPaints = false;
        this.lastStroke = this.node.getStroke();
      };

      proto.disposePaintableState = function() {
        this.fillObserver.clean();
        this.strokeObserver.clean();
      };

      proto.markDirtyFill = function() {
        this.dirtyFill = true;
        this.markPaintDirty();
        this.fillObserver.update(); // TODO: look into having the fillObserver be notified of Node changes as our source
      };

      proto.markDirtyStroke = function() {
        this.dirtyStroke = true;
        this.markPaintDirty();
        this.strokeObserver.update(); // TODO: look into having the strokeObserver be notified of Node changes as our source
      };

      proto.markDirtyLineWidth = function() {
        this.dirtyLineWidth = true;
        this.markPaintDirty();
      };

      proto.markDirtyLineOptions = function() {
        this.dirtyLineOptions = true;
        this.markPaintDirty();
      };

      proto.markDirtyCachedPaints = function() {
        this.dirtyCachedPaints = true;
        this.markPaintDirty();
      };
    }
  };

  // mix-in for Canvas drawables
  Paintable.PaintableStatelessDrawable = {
    mixin: function PaintableStatelessDrawable( drawableType ) {
      var proto = drawableType.prototype;

      proto.initializePaintableStateless = function( renderer, instance ) {
        this.fillCallback = this.fillCallback || this.markDirtyFill.bind( this );
        this.strokeCallback = this.strokeCallback || this.markDirtyStroke.bind( this );
        this.fillObserver = this.fillObserver || new Paintable.PaintObserver( 'fill', this.fillCallback );
        this.strokeObserver = this.strokeObserver || new Paintable.PaintObserver( 'stroke', this.strokeCallback );

        this.fillObserver.initialize( instance.node );
        this.strokeObserver.initialize( instance.node );

        return this;
      };

      proto.disposePaintableStateless = function() {
        this.fillObserver.clean();
        this.strokeObserver.clean();
      };

      proto.markDirtyFill = function() {
        this.markPaintDirty();
        this.fillObserver.update(); // TODO: look into having the fillObserver be notified of Node changes as our source
      };

      proto.markDirtyStroke = function() {
        this.markPaintDirty();
        this.strokeObserver.update(); // TODO: look into having the strokeObserver be notified of Node changes as our source
      };

      proto.markDirtyLineWidth = function() {
        this.markPaintDirty();
      };

      proto.markDirtyLineOptions = function() {
        this.markPaintDirty();
      };

      proto.markDirtyCachedPaints = function() {
        this.markPaintDirty();
      };
    }
  };

  /**
   * Returns the SVG style string used to represent a paint.
   *
   * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} paint
   * @param {SVGBlock} svgBlock
   */
  function paintToSVGStyle( paint, svgBlock ) {
    if ( !paint ) {
      // no paint
      return 'none';
    }
    else if ( paint.toCSS ) {
      // Color object paint
      return paint.toCSS();
    }
    else if ( paint.isPaint ) {
      // reference the SVG definition with a URL
      return 'url(#' + paint.id + '-' + ( svgBlock ? svgBlock.id : 'noblock' ) + ')';
    }
    else {
      // plain CSS color
      return paint;
    }
  }

  // handles SVG defs and fill/stroke style for SVG elements (by composition, not a mix-in or for inheritance)
  Paintable.PaintSVGState = function PaintSVGState() {
    this.initialize();
  };
  inherit( Object, Paintable.PaintSVGState, {
    initialize: function() {
      this.svgBlock = null; // {SVGBlock | null}

      // {string} fill/stroke style fragments that are currently used
      this.fillStyle = 'none';
      this.strokeStyle = 'none';

      // current reference-counted fill/stroke paints (gradients and fills) that will need to be released on changes
      // or disposal
      this.fillPaint = null;
      this.strokePaint = null;

      // these are used by the actual SVG element
      this.updateBaseStyle(); // the main style CSS
      this.strokeDetailStyle = ''; // width/dash/cap/join CSS
    },

    dispose: function() {
      // be cautious, release references
      this.releaseFillPaint();
      this.releaseStrokePaint();
    },

    releaseFillPaint: function() {
      if ( this.fillPaint ) {
        this.svgBlock.decrementPaint( this.fillPaint );
        this.fillPaint = null;
      }
    },

    releaseStrokePaint: function() {
      if ( this.strokePaint ) {
        this.svgBlock.decrementPaint( this.strokePaint );
        this.strokePaint = null;
      }
    },

    /**
     * Called when the fill needs to be updated, with the latest defs SVG block
     * @public (scenery-internal)
     *
     * @param {SVGBlock} svgBlock
     * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} fill
     */
    updateFill: function( svgBlock, fill ) {
      assert && assert( this.svgBlock === svgBlock );

      // NOTE: If fill.isPaint === true, this should be different if we switched to a different SVG block.
      var fillStyle = paintToSVGStyle( fill, svgBlock );

      // If our fill paint reference changed
      if ( fill !== this.fillPaint ) {
        // release the old reference
        this.releaseFillPaint();

        // only store a new reference if our new fill is a paint
        if ( fill && fill.isPaint ) {
          this.fillPaint = fill;
          svgBlock.incrementPaint( fill );
        }
      }

      // If we need to update the SVG style of our fill
      if ( fillStyle !== this.fillStyle ) {
        this.fillStyle = fillStyle;
        this.updateBaseStyle();
      }
    },

    /**
     * Called when the stroke needs to be updated, with the latest defs SVG block
     * @public (scenery-internal)
     *
     * @param {SVGBlock} svgBlock
     * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} fill
     */
    updateStroke: function( svgBlock, stroke ) {
      assert && assert( this.svgBlock === svgBlock );

      // NOTE: If stroke.isPaint === true, this should be different if we switched to a different SVG block.
      var strokeStyle = paintToSVGStyle( stroke, svgBlock );

      // If our stroke paint reference changed
      if ( stroke !== this.strokePaint ) {
        // release the old reference
        this.releaseStrokePaint();

        // only store a new reference if our new stroke is a paint
        if ( stroke && stroke.isPaint ) {
          this.strokePaint = stroke;
          svgBlock.incrementPaint( stroke );
        }
      }

      // If we need to update the SVG style of our stroke
      if ( strokeStyle !== this.strokeStyle ) {
        this.strokeStyle = strokeStyle;
        this.updateBaseStyle();
      }
    },

    updateBaseStyle: function() {
      this.baseStyle = 'fill: ' + this.fillStyle + '; stroke: ' + this.strokeStyle + ';';
    },

    updateStrokeDetailStyle: function( node ) {
      var strokeDetailStyle = '';

      var lineWidth = node.getLineWidth();
      if ( lineWidth !== 1 ) {
        strokeDetailStyle += 'stroke-width: ' + lineWidth + ';';
      }

      var lineCap = node.getLineCap();
      if ( lineCap !== 'butt' ) {
        strokeDetailStyle += 'stroke-linecap: ' + lineCap + ';';
      }

      var lineJoin = node.getLineJoin();
      if ( lineJoin !== 'miter' ) {
        strokeDetailStyle += 'stroke-linejoin: ' + lineJoin + ';';
      }

      var miterLimit = node.getMiterLimit();
      strokeDetailStyle += 'stroke-miterlimit: ' + miterLimit + ';';

      if ( node.hasLineDash() ) {
        strokeDetailStyle += 'stroke-dasharray: ' + node.getLineDash().join( ',' ) + ';';
        strokeDetailStyle += 'stroke-dashoffset: ' + node.getLineDashOffset() + ';';
      }

      this.strokeDetailStyle = strokeDetailStyle;
    },

    // called when the defs SVG block is switched (our SVG element was moved to another SVG top-level context)
    updateSVGBlock: function( svgBlock ) {
      // remove paints from the old svgBlock
      var oldSvgBlock = this.svgBlock;
      if ( oldSvgBlock ) {
        if ( this.fillPaint ) {
          oldSvgBlock.decrementPaint( this.fillPaint );
        }
        if ( this.strokePaint ) {
          oldSvgBlock.decrementPaint( this.strokePaint );
        }
      }

      this.svgBlock = svgBlock;

      // add paints to the new svgBlock
      if ( this.fillPaint ) {
        svgBlock.incrementPaint( this.fillPaint );
      }
      if ( this.strokePaint ) {
        svgBlock.incrementPaint( this.strokePaint );
      }
    }
  } );

  return Paintable;
} );