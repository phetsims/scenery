// Copyright 2002-2014, University of Colorado Boulder

/**
 * Mix-in for nodes that support a standard fill and/or stroke.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var LineStyles = require( 'KITE/util/LineStyles' );

  var inherit = require( 'PHET_CORE/inherit' );
  var extend = require( 'PHET_CORE/extend' );
  var platform = require( 'PHET_CORE/platform' );

  var isSafari5 = platform.safari5;
  var isIE9 = platform.ie9;

  scenery.Paintable = function Paintable( type ) {
    var proto = type.prototype;

    extend( proto, {
      // this should be called in the constructor to initialize
      initializePaintable: function() {
        this._fill = null;
        this._stroke = null;
        this._fillPickable = true;
        this._strokePickable = false;
        this._fillKept = false; // whether to keep SVG defs for gradients/fills around to improve performance
        this._strokeKept = false; // whether the SVG stroke should be kept (makes gradients/patterns stay in memory!)
        this._lineDrawingStyles = new LineStyles();

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

      setFill: function( fill ) {
        if ( this.getFill() !== fill ) {
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

          var stateLen = this._drawables.length;
          for ( var i = 0; i < stateLen; i++ ) {
            this._drawables[i].markDirtyFill();
          }
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

      isFillKept: function() {
        return this._fillKept;
      },

      setFillKept: function( kept ) {
        assert && assert( typeof kept === 'boolean' );

        this._fillKept = kept;

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
            this._drawables[i].markDirtyLineWidth();
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
            this._drawables[i].markDirtyLineOptions();
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
            this._drawables[i].markDirtyLineOptions();
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
            this._drawables[i].markDirtyLineOptions();
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
            this._drawables[i].markDirtyLineOptions();
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
            this._drawables[i].markDirtyLineOptions();
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

      isStrokeKept: function() {
        return this._strokeKept;
      },

      setStrokeKept: function( kept ) {
        assert && assert( typeof kept === 'boolean' );

        this._strokeKept = kept;

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

      setStroke: function( stroke ) {
        if ( this.getStroke() !== stroke ) {

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

          var stateLen = this._drawables.length;
          for ( var i = 0; i < stateLen; i++ ) {
            this._drawables[i].markDirtyStroke();
          }
        }
        return this;
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

      // if we have to apply a transform workaround for https://github.com/phetsims/scenery/issues/196 (only when we have a pattern or gradient)
      requiresSVGBoundsWorkaround: function() {
        if ( !this._stroke || !this._stroke.getSVGDefinition ) {
          return false;
        }

        var bounds = this.computeShapeBounds( false ); // without stroke
        return bounds.x * bounds.y === 0; // at least one of them was zero, so the bounding box has no area
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
            if ( self[prop] !== defaultStyles[prop] ) {
              addProp( prop, self[prop] );
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
          bitmask |= scenery.bitmaskSupportsSVG;
        }

        // we always have Canvas support?
        bitmask |= scenery.bitmaskSupportsCanvas;

        // nothing in the fill can change whether its bounds are valid
        bitmask |= scenery.bitmaskBoundsValid;

        if ( !this._fill ) {
          // if there is no fill, it is supported by DOM
          bitmask |= scenery.bitmaskSupportsDOM;
        }
        else if ( this._fill.isPattern ) {
          // no pattern support for DOM (for now!)
        }
        else if ( this._fill.isGradient ) {
          // no gradient support for DOM (for now!)
        }
        else {
          // solid fills always supported for DOM
          bitmask |= scenery.bitmaskSupportsDOM;
        }

        return bitmask;
      },

      getStrokeRendererBitmask: function() {
        var bitmask = 0;

        if ( !( isIE9 && this.hasStroke() && this.hasLineDash() ) ) {
          bitmask |= scenery.bitmaskSupportsCanvas;
        }

        // always have SVG support (for now?)
        bitmask |= scenery.bitmaskSupportsSVG;

        // for now, nothing about the stroke prevents us from having valid bounds (we compute these offsets)
        bitmask |= scenery.bitmaskBoundsValid;

        if ( !this.hasStroke() ) {
          // allow DOM support if there is no stroke
          bitmask |= scenery.bitmaskSupportsDOM;
        }

        return bitmask;
      }
    } );

    // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
    proto._mutatorKeys = [
      'fill', 'fillPickable', 'fillKept', 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'miterLimit', 'lineDash',
      'lineDashOffset', 'strokePickable', 'strokeKept'
    ].concat( proto._mutatorKeys );

    Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
    Object.defineProperty( proto, 'fillPickable', { set: proto.setFillPickable, get: proto.isFillPickable } );
    Object.defineProperty( proto, 'fillKept', { set: proto.setFillKept, get: proto.isFillKept } );
    Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
    Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
    Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
    Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
    Object.defineProperty( proto, 'miterLimit', { set: proto.setMiterLimit, get: proto.getMiterLimit } );
    Object.defineProperty( proto, 'lineDash', { set: proto.setLineDash, get: proto.getLineDash } );
    Object.defineProperty( proto, 'lineDashOffset', { set: proto.setLineDashOffset, get: proto.getLineDashOffset } );
    Object.defineProperty( proto, 'strokePickable', { set: proto.setStrokePickable, get: proto.isStrokePickable } );
    Object.defineProperty( proto, 'strokeKept', { set: proto.setStrokeKept, get: proto.isStrokeKept } );

    if ( proto.invalidateFill ) {
      var oldInvalidateFill = proto.invalidateFill;
      proto.invalidateFill = function() {
        this.invalidateSupportedRenderers();
        oldInvalidateFill.call( this );
      };
    }
    else {
      proto.invalidateFill = function() {
        this.invalidateSupportedRenderers();
      };
    }

    if ( proto.invalidateStroke ) {
      var oldInvalidateStroke = proto.invalidateStroke;
      proto.invalidateStroke = function() {
        this.invalidateSupportedRenderers();
        oldInvalidateStroke.call( this );
      };
    }
    else {
      proto.invalidateStroke = function() {
        this.invalidateSupportedRenderers();
      };
    }
  };
  var Paintable = scenery.Paintable;

  // mix-in base for DOM and SVG drawables
  // NOTE: requires state.node to be defined
  Paintable.PaintableState = function PaintableState( drawableType ) {
    var proto = drawableType.prototype;

    proto.initializePaintableState = function() {
      this.lastFill = undefined;
      this.dirtyFill = true;

      this.lastStroke = undefined;
      this.dirtyStroke = true;
      this.dirtyLineWidth = true;
      this.dirtyLineOptions = true; // e.g. cap, join, dash, dashoffset, miterlimit
    };

    proto.cleanPaintableState = function() {
      this.dirtyFill = false;
      this.lastFill = this.node.getFill();

      this.dirtyStroke = false;
      this.dirtyLineWidth = false;
      this.dirtyLineOptions = false;
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
  };

  // mix-in for Canvas drawables
  Paintable.PaintableStateless = function PaintableStateless( drawableType ) {
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
  };

  var fillableSVGIdCounter = 0;
  var strokableSVGIdCounter = 0;

  // handles SVG defs and fill/stroke style for SVG elements (by composition, not a mix-in or for inheritance)
  Paintable.PaintSVGState = function PaintSVGState() {
    this.fillId = 'svgfill' + ( fillableSVGIdCounter++ );
    this.strokeId = 'svgstroke' + ( strokableSVGIdCounter++ );

    this.initialize();
  };
  inherit( Object, Paintable.PaintSVGState, {
    initialize: function() {
      this.fill = null;
      this.stroke = null;
      this.fillDef = null;
      this.strokeDef = null;

      // these are used by the actual SVG element
      this.baseStyle = this.computeStyle(); // the main style CSS
      this.extraStyle = '';                 // width/dash/cap/join CSS
    },

    dispose: function() {
      // be cautious, release references
      this.fill = null;
      this.stroke = null;
      this.releaseFillDef();
      this.releaseStrokeDef();
    },

    releaseFillDef: function() {
      if ( this.fillDef ) {
        this.fillDef.parentNode.removeChild( this.fillDef );
        this.fillDef = null;
      }
    },

    releaseStrokeDef: function() {
      if ( this.strokeDef ) {
        this.strokeDef.parentNode.removeChild( this.strokeDef );
        this.strokeDef = null;
      }
    },

    // called when the fill needs to be updated, with the latest defs SVG block
    updateFill: function( svgBlock, fill ) {
      if ( fill !== this.fill ) {
        this.releaseFillDef();
        this.fill = fill;
        this.baseStyle = this.computeStyle();
        if ( this.fill && this.fill.getSVGDefinition ) {
          this.fillDef = this.fill.getSVGDefinition( this.fillId );
          svgBlock.defs.appendChild( this.fillDef );
        }
      }
    },

    updateStroke: function( svgBlock, stroke ) {
      if ( stroke !== this.stroke ) {
        this.releaseStrokeDef();
        this.stroke = stroke;
        this.baseStyle = this.computeStyle();
        if ( this.stroke && this.stroke.getSVGDefinition ) {
          this.strokeDef = this.stroke.getSVGDefinition( this.strokeId );
          svgBlock.defs.appendChild( this.strokeDef );
        }
      }
    },

    updateStrokeParameters: function( node ) {
      var extraStyle = '';

      var lineWidth = node.getLineWidth();
      if ( lineWidth !== 1 ) {
        extraStyle += 'stroke-width: ' + lineWidth + ';';
      }

      var lineCap = node.getLineCap();
      if ( lineCap !== 'butt' ) {
        extraStyle += 'stroke-linecap: ' + lineCap + ';';
      }

      var lineJoin = node.getLineJoin();
      if ( lineJoin !== 'miter' ) {
        extraStyle += 'stroke-linejoin: ' + lineJoin + ';';
      }

      var miterLimit = node.getMiterLimit();
      extraStyle += 'stroke-miterlimit: ' + miterLimit + ';';

      if ( node.hasLineDash() ) {
        extraStyle += 'stroke-dasharray: ' + node.getLineDash().join( ',' ) + ';';
        extraStyle += 'stroke-dashoffset: ' + node.getLineDashOffset() + ';';
      }

      this.extraStyle = extraStyle;
    },

    // called when the defs SVG block is switched (our SVG element was moved to another SVG top-level context)
    updateSVGBlock: function( svgBlock ) {
      if ( this.fillDef ) {
        if ( svgBlock.defs ) {
          // adding to the DOM here removes it from its previous location
          svgBlock.defs.appendChild( this.fillDef );
        }
        else if ( this.fillDef.parentNode ) {
          //OHTWO TODO: does this parentNode access cause reflows?
          this.fillDef.parentNode.removeChild( this.fillDef );
        }
      }
      if ( this.strokeDef ) {
        if ( svgBlock.defs ) {
          // adding to the DOM here removes it from its previous location
          svgBlock.defs.appendChild( this.strokeDef );
        }
        else if ( this.strokeDef.parentNode ) {
          //OHTWO TODO: does this parentNode access cause reflows?
          this.strokeDef.parentNode.removeChild( this.strokeDef );
        }
      }
    },

    computeStyle: function() {
      var style = 'fill: ';
      if ( !this.fill ) {
        // no fill
        style += 'none;';
      }
      else if ( this.fill.toCSS ) {
        // Color object fill
        style += this.fill.toCSS() + ';';
      }
      else if ( this.fill.getSVGDefinition ) {
        // reference the SVG definition with a URL
        style += 'url(#' + this.fillId + ');';
      }
      else {
        // plain CSS color
        style += this.fill + ';';
      }

      if ( !this.stroke ) {
        // no stroke
        style += ' stroke: none;';
      } else {
        style += ' stroke: ';
        if ( this.stroke.toCSS ) {
          // Color object stroke
          style += this.stroke.toCSS() + ';';
        }
        else if ( this.stroke.getSVGDefinition ) {
          // reference the SVG definition with a URL
          style += 'url(#' + this.strokeId + ');';
        }
        else {
          // plain CSS color
          style += this.stroke + ';';
        }
      }

      return style;
    }
  } );

  return Paintable;
} );