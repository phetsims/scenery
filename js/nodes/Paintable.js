// Copyright 2002-2014, University of Colorado Boulder

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

  var isSafari5 = platform.safari5;
  var isIE9 = platform.ie9;

  /*
   * Applies the mix-in to a subtype of Node.
   *
   * @param {constructor} type - A constructor that inherits from Node
   */
  scenery.Paintable = {
    mixin: function( type ) {
      var proto = type.prototype;

      extend( proto, {
        // this should be called in the constructor to initialize
        initializePaintable: function() {
          this._fill = null;
          this._fillPickable = true;

          this._stroke = null;
          this._strokePickable = false;

          this._cachedPaints = [];
          this._lineDrawingStyles = new LineStyles();

          this._fillColor = null;
          this._fillColorDirty = true;
          this._strokeColor = null;
          this._strokeColorDirty = true;

          var that = this;
          this._fillListener = function() {
            that.invalidateFill();
          };
          this._strokeListener = function() {
            that.invalidateStroke();
          };
        },

        hasFill: function() {
          return this._fill !== null;
        },

        getFill: function() {
          return this._fill;
        },

        validateFillColor: function() {
          if ( this._fillColorDirty ) {
            this._fillColorDirty = false;

            if ( typeof this._fill === 'string' || this._fill instanceof Color ) {
              if ( this._fillColor ) {
                this._fillColor.set( this._fill );
              }
              // lazily create a Color when necessary, instead of pre-allocating
              else {
                this._fillColor = new Color( this._fill );
              }
            }
          }
        },

        /**
         * If the current fill is a solid color (string or scenery.Color), getFillColor() will return a scenery.Color
         * reference. This reference should be considered immutable (should not be modified)
         *
         * @returns {Color | null} [read-only]
         */
        getFillColor: function() {
          this.validateFillColor();

          // types of fills where we can return a single color
          if ( typeof this._fill === 'string' || this._fill instanceof Color ) {
            return this._fillColor;
          }
          // no fill, or a pattern/gradient (we can't return a single fill)
          else {
            return null;
          }
        },

        setFill: function( fill ) {
          // Instance equality used here since it would be more expensive to parse all CSS
          // colors and compare every time the fill changes. Right now, usually we don't have
          // to parse CSS colors. See https://github.com/phetsims/scenery/issues/255
          if ( this._fill !== fill ) {
            this._fillColorDirty = true;

            //OHTWO TODO: we probably shouldn't be checking this here?
            var hasInstances = this._instances.length > 0;

            if ( hasInstances && this._fill && this._fill.removeChangeListener ) {
              this._fill.removeChangeListener( this._fillListener );
            }

            this._fill = fill;

            if ( hasInstances && this._fill && this._fill.addChangeListener ) {
              this._fill.addChangeListener( this._fillListener );
            }

            this.invalidateFill();
          }
          return this;
        },

        isFillPickable: function() {
          return this._fillPickable;
        },

        setFillPickable: function( pickable ) {
          assert && assert( typeof pickable === 'boolean' );
          if ( this._fillPickable !== pickable ) {
            this._fillPickable = pickable;

            // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
            this.invalidateFill();
          }
          return this;
        },

        hasStroke: function() {
          return this._stroke !== null;
        },

        // TODO: setting these properties looks like a good candidate for refactoring to lessen file size
        getLineWidth: function() {
          return this._lineDrawingStyles.lineWidth;
        },

        setLineWidth: function( lineWidth ) {
          assert && assert( typeof lineWidth === 'number', 'lineWidth should be a number, not ' + lineWidth );

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

        getLineCap: function() {
          return this._lineDrawingStyles.lineCap;
        },

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

        getLineJoin: function() {
          return this._lineDrawingStyles.lineJoin;
        },

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

        getMiterLimit: function() {
          return this._lineDrawingStyles.miterLimit;
        },

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

        getLineDash: function() {
          return this._lineDrawingStyles.lineDash;
        },

        hasLineDash: function() {
          return !!this._lineDrawingStyles.lineDash.length;
        },

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

        getLineDashOffset: function() {
          return this._lineDrawingStyles.lineDashOffset;
        },

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

        isStrokePickable: function() {
          return this._strokePickable;
        },

        setStrokePickable: function( pickable ) {
          assert && assert( typeof pickable === 'boolean', 'strokePickable should be a boolean, not ' + pickable );

          if ( this._strokePickable !== pickable ) {
            this._strokePickable = pickable;

            // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
            this.invalidateStroke();
          }
          return this;
        },

        setLineStyles: function( lineStyles ) {

          this._lineDrawingStyles = lineStyles;
          this.invalidateStroke();
          return this;
        },

        getLineStyles: function() {
          return this._lineDrawingStyles;
        },

        getStroke: function() {
          return this._stroke;
        },

        validateStrokeColor: function() {
          if ( this._strokeColorDirty ) {
            this._strokeColorDirty = false;

            if ( typeof this._stroke === 'string' || this._stroke instanceof Color ) {
              if ( this._strokeColor ) {
                this._strokeColor.set( this._stroke );
              }
              // lazily create a Color when necessary, instead of pre-allocating
              else {
                this._strokeColor = new Color( this._stroke );
              }
            }
          }
        },

        /**
         * If the current stroke is a solid color (string or scenery.Color), getStrokeColor() will return a scenery.Color
         * reference. This reference should be considered immutable (should not be modified)
         *
         * @returns {Color | null} [read-only]
         */
        getStrokeColor: function() {
          this.validateStrokeColor();

          // types of strokes where we can return a single color
          if ( typeof this._stroke === 'string' || this._stroke instanceof Color ) {
            return this._strokeColor;
          }
          // no stroke, or a pattern/gradient (we can't return a single stroke)
          else {
            return null;
          }
        },

        setStroke: function( stroke ) {
          if ( this._stroke !== stroke ) {
            this._strokeColorDirty = true;

            //OHTWO TODO: probably shouldn't have a reference here
            var hasInstances = this._instances.length > 0;

            if ( hasInstances && this._stroke && this._stroke.removeChangeListener ) {
              this._stroke.removeChangeListener( this._strokeListener );
            }

            this._stroke = stroke;

            if ( hasInstances && this._stroke && this._stroke.addChangeListener ) {
              this._stroke.addChangeListener( this._strokeListener );
            }

            this.invalidateStroke();
          }
          return this;
        },

        getCachedPaints: function() {
          return this._cachedPaints;
        },

        /*
         * Sets the cached paints to the input array (a defensive copy).
         *
         * @param {Paint[]} paints
         */
        setCachedPaints: function( paints ) {
          this._cachedPaints = paints.filter( function( paint ) { return paint && paint.isPaint; } );

          var stateLen = this._drawables.length;
          for ( var i = 0; i < stateLen; i++ ) {
            this._drawables[ i ].markDirtyCachedPaints();
          }

          return this;
        },

        addCachedPaint: function( paint ) {
          if ( paint && paint.isPaint ) {
            this._cachedPaints.push( paint );

            var stateLen = this._drawables.length;
            for ( var i = 0; i < stateLen; i++ ) {
              this._drawables[ i ].markDirtyCachedPaints();
            }
          }
        },

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

        firstInstanceAdded: function() {
          if ( this._fill && this._fill.addChangeListener ) {
            this._fill.addChangeListener( this._fillListener );
          }
          if ( this._stroke && this._stroke.addChangeListener ) {
            this._stroke.addChangeListener( this._strokeListener );
          }
        },

        lastInstanceRemoved: function() {
          if ( this._fill && this._fill.removeChangeListener ) {
            this._fill.removeChangeListener( this._fillListener );
          }
          if ( this._stroke && this._stroke.removeChangeListener ) {
            this._stroke.removeChangeListener( this._strokeListener );
          }
        },

        beforeCanvasFill: function( wrapper ) {
          wrapper.setFillStyle( this._fill );
          if ( this._fill.transformMatrix ) {
            wrapper.context.save();
            this._fill.transformMatrix.canvasAppendTransform( wrapper.context );
          }
        },

        afterCanvasFill: function( wrapper ) {
          if ( this._fill.transformMatrix ) {
            wrapper.context.restore();
          }
        },

        beforeCanvasStroke: function( wrapper ) {
          // TODO: is there a better way of not calling so many things on each stroke?
          wrapper.setStrokeStyle( this._stroke );
          wrapper.setLineWidth( this.getLineWidth() );
          wrapper.setLineCap( this.getLineCap() );
          wrapper.setLineJoin( this.getLineJoin() );
          wrapper.setMiterLimit( this.getMiterLimit() );
          wrapper.setLineDash( this.getLineDash() );
          wrapper.setLineDashOffset( this.getLineDashOffset() );
          if ( this._stroke.transformMatrix ) {
            wrapper.context.save();
            this._stroke.transformMatrix.canvasAppendTransform( wrapper.context );
          }
        },

        afterCanvasStroke: function( wrapper ) {
          if ( this._stroke.transformMatrix ) {
            wrapper.context.restore();
          }
        },

        getCSSFill: function() {
          // if it's a Color object, get the corresponding CSS
          // 'transparent' will make us invisible if the fill is null
          return this._fill ? ( this._fill.toCSS ? this._fill.toCSS() : this._fill ) : 'transparent';
        },

        getSimpleCSSStroke: function() {
          // if it's a Color object, get the corresponding CSS
          // 'transparent' will make us invisible if the fill is null
          return this._stroke ? ( this._stroke.toCSS ? this._stroke.toCSS() : this._stroke ) : 'transparent';
        },

        appendFillablePropString: function( spaces, result ) {
          if ( this._fill ) {
            if ( result ) {
              result += ',\n';
            }
            if ( typeof this._fill === 'string' ) {
              result += spaces + 'fill: \'' + this._fill + '\'';
            }
            else {
              result += spaces + 'fill: ' + this._fill.toString();
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
            if ( typeof this._stroke === 'string' ) {
              addProp( 'stroke', this._stroke );
            }
            else {
              addProp( 'stroke', this._stroke.toString(), true );
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

        getFillRendererBitmask: function() {
          var bitmask = 0;

          // Safari 5 has buggy issues with SVG gradients
          if ( !( isSafari5 && this._fill && this._fill.isGradient ) ) {
            bitmask |= Renderer.bitmaskSVG;
          }

          // we always have Canvas support?
          bitmask |= Renderer.bitmaskCanvas;

          if ( !this._fill ) {
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
            // solid fills always supported for DOM, WebGL and Pixi
            bitmask |= Renderer.bitmaskDOM;
          }
          bitmask |= Renderer.bitmaskPixi;

          return bitmask;
        },

        getStrokeRendererBitmask: function() {
          var bitmask = 0;

          if ( !( isIE9 && this.hasStroke() && this.hasLineDash() ) ) {
            bitmask |= Renderer.bitmaskCanvas;
          }

          // always have SVG support (for now?)
          bitmask |= Renderer.bitmaskSVG;

          if ( !this.hasStroke() ) {
            // allow DOM support if there is no stroke
            bitmask |= Renderer.bitmaskDOM;
          }

          bitmask |= Renderer.bitmaskPixi;

          return bitmask;
        }
      } );

      // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
      proto._mutatorKeys = [
        'fill', 'fillPickable', 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'miterLimit', 'lineDash',
        'lineDashOffset', 'strokePickable', 'cachedPaints'
      ].concat( proto._mutatorKeys );

      Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
      Object.defineProperty( proto, 'fillColor', { set: proto.setFill, get: proto.getFillColor } );
      Object.defineProperty( proto, 'fillPickable', { set: proto.setFillPickable, get: proto.isFillPickable } );
      Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
      Object.defineProperty( proto, 'strokeColor', { set: proto.setStroke, get: proto.getStrokeColor } );
      Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
      Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
      Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
      Object.defineProperty( proto, 'miterLimit', { set: proto.setMiterLimit, get: proto.getMiterLimit } );
      Object.defineProperty( proto, 'lineDash', { set: proto.setLineDash, get: proto.getLineDash } );
      Object.defineProperty( proto, 'lineDashOffset', { set: proto.setLineDashOffset, get: proto.getLineDashOffset } );
      Object.defineProperty( proto, 'strokePickable', { set: proto.setStrokePickable, get: proto.isStrokePickable } );
      Object.defineProperty( proto, 'cachedPaints', { set: proto.setCachedPaints, get: proto.getCachedPaints } );

      // Paintable's version of invalidateFill()
      function invalidateFill() {
        /*jshint -W040 */
        this.invalidateSupportedRenderers();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyFill();
        }
        /*jshint +W040 */
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
        /*jshint -W040 */
        this.invalidateSupportedRenderers();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyStroke();
        }
        /*jshint +W040 */
      }
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
  var Paintable = scenery.Paintable;

  // mix-in base for DOM and SVG drawables
  // NOTE: requires state.node to be defined
  Paintable.PaintableStatefulDrawable = {
    mixin: function PaintableStatefulDrawable( drawableType ) {
      var proto = drawableType.prototype;

      proto.initializePaintableState = function() {
        this.lastFill = undefined;
        this.dirtyFill = true;

        this.lastStroke = undefined;
        this.dirtyStroke = true;
        this.dirtyLineWidth = true;
        this.dirtyLineOptions = true; // e.g. cap, join, dash, dashoffset, miterlimit
        this.dirtyCachedPaints = true;
        this.lastCachedPaints = [];
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

      proto.markDirtyFill = function() {
        this.dirtyFill = true;
        this.markPaintDirty();
      };

      proto.markDirtyStroke = function() {
        this.dirtyStroke = true;
        this.markPaintDirty();
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

      proto.markDirtyFill = function() {
        this.markPaintDirty();
      };

      proto.markDirtyStroke = function() {
        this.markPaintDirty();
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

    // called when the fill needs to be updated, with the latest defs SVG block
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