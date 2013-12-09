// Copyright 2002-2013, University of Colorado

/**
 * Mix-in for nodes that support a standard stroke.
 *
 * TODO: miterLimit handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  
  var platform = require( 'PHET_CORE/platform' );
  
  var isIE9 = platform.ie9;
  
  scenery.Strokable = function Strokable( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeStrokable = function() {
      this._stroke = null;
      this._strokePickable = false;
      this._lineDrawingStyles = new LineStyles();
      
      var that = this;
      this._strokeListener = function() {
        that.invalidatePaint(); // TODO: move this to invalidateStroke?
        that.invalidateStroke();
      };
    };
    
    proto.hasStroke = function() {
      return this._stroke !== null;
    };
    
    // TODO: setting these properties looks like a good candidate for refactoring to lessen file size
    proto.getLineWidth = function() {
      return this._lineDrawingStyles.lineWidth;
    };
    
    proto.setLineWidth = function( lineWidth ) {
      if ( this.getLineWidth() !== lineWidth ) {
        this.markOldSelfPaint(); // since the previous line width may have been wider
        
        this._lineDrawingStyles.lineWidth = lineWidth;
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyLineWidth();
        }
      }
      return this;
    };
    
    proto.getLineCap = function() {
      return this._lineDrawingStyles.lineCap;
    };
    
    proto.setLineCap = function( lineCap ) {
      if ( this._lineDrawingStyles.lineCap !== lineCap ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineCap = lineCap;
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyLineOptions();
        }
      }
      return this;
    };
    
    proto.getLineJoin = function() {
      return this._lineDrawingStyles.lineJoin;
    };
    
    proto.setLineJoin = function( lineJoin ) {
      if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineJoin = lineJoin;
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyLineOptions();
        }
      }
      return this;
    };
    
    proto.getLineDash = function() {
      return this._lineDrawingStyles.lineDash;
    };
    
    proto.hasLineDash = function() {
      return !!this._lineDrawingStyles.lineDash.length;
    };
    
    proto.setLineDash = function( lineDash ) {
      if ( this._lineDrawingStyles.lineDash !== lineDash ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineDash = lineDash || [];
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyLineOptions();
        }
      }
      return this;
    };
    
    proto.getLineDashOffset = function() {
      return this._lineDrawingStyles.lineDashOffset;
    };
    
    proto.setLineDashOffset = function( lineDashOffset ) {
      if ( this._lineDrawingStyles.lineDashOffset !== lineDashOffset ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineDashOffset = lineDashOffset;
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyLineOptions();
        }
      }
      return this;
    };
    
    proto.isStrokePickable = function() {
      return this._strokePickable;
    };
    
    proto.setStrokePickable = function( pickable ) {
      assert && assert( typeof pickable === 'boolean' );
      if ( this._strokePickable !== pickable ) {
        this._strokePickable = pickable;
        
        // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.setLineStyles = function( lineStyles ) {
      // TODO: since we have been using lineStyles as mutable for now, lack of change check is good here?
      this.markOldSelfPaint();
      
      this._lineDrawingStyles = lineStyles;
      this.invalidateStroke();
      return this;
    };
    
    proto.getLineStyles = function() {
      return this._lineDrawingStyles;
    };
    
    proto.getStroke = function() {
      return this._stroke;
    };
    
    proto.setStroke = function( stroke ) {
      if ( this.getStroke() !== stroke ) {
        // since this can actually change the bounds, we need to handle a few things differently than the fill
        this.markOldSelfPaint();
        
        var hasInstances = this._instances.length > 0;
        
        if ( hasInstances && this._stroke && this._stroke.removeChangeListener ) {
          this._stroke.removeChangeListener( this._strokeListener );
        }
        
        this._stroke = stroke;
        
        if ( hasInstances && this._stroke && this._stroke.addChangeListener ) {
          this._stroke.addChangeListener( this._strokeListener );
        }
        
        this.invalidateStroke();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates[i].markDirtyStroke();
        }
      }
      return this;
    };
    
    var superFirstInstanceAdded = proto.firstInstanceAdded;
    proto.firstInstanceAdded = function() {
      if ( this._stroke && this._stroke.addChangeListener ) {
        this._stroke.addChangeListener( this._strokeListener );
      }
      
      if ( superFirstInstanceAdded ) {
        superFirstInstanceAdded.call( this );
      }
    };
    
    var superLastInstanceRemoved = proto.lastInstanceRemoved;
    proto.lastInstanceRemoved = function() {
      if ( this._stroke && this._stroke.removeChangeListener ) {
        this._stroke.removeChangeListener( this._strokeListener );
      }
      
      if ( superLastInstanceRemoved ) {
        superLastInstanceRemoved.call( this );
      }
    };
    
    proto.beforeCanvasStroke = function( wrapper ) {
      // TODO: is there a better way of not calling so many things on each stroke?
      wrapper.setStrokeStyle( this._stroke );
      wrapper.setLineWidth( this.getLineWidth() );
      wrapper.setLineCap( this.getLineCap() );
      wrapper.setLineJoin( this.getLineJoin() );
      wrapper.setLineDash( this.getLineDash() );
      wrapper.setLineDashOffset( this.getLineDashOffset() );
      if ( this._stroke.transformMatrix ) {
        wrapper.context.save();
        this._stroke.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    };
    
    proto.afterCanvasStroke = function( wrapper ) {
      if ( this._stroke.transformMatrix ) {
        wrapper.context.restore();
      }
    };
    
    proto.getSVGStrokeStyle = function() {
      if ( !this._stroke ) {
        // no stroke
        return 'stroke: none;';
      }
      
      var style = 'stroke: ';
      if ( this._stroke.toCSS ) {
        // Color object stroke
        style += this._stroke.toCSS() + ';';
      } else if ( this._stroke.getSVGDefinition ) {
        // reference the SVG definition with a URL
        style += 'url(#stroke' + this.getId() + ');';
      } else {
        // plain CSS color
        style += this._stroke + ';';
      }
      
      // TODO: don't include unnecessary directives? - is it worth any branching cost?
      style += 'stroke-width: ' + this.getLineWidth() + ';';
      style += 'stroke-linecap: ' + this.getLineCap() + ';';
      style += 'stroke-linejoin: ' + this.getLineJoin() + ';';
      if ( this.hasLineDash() ) {
        style += 'stroke-dasharray: ' + this.getLineDash().join( ',' ) + ';';
        style += 'stroke-dashoffset: ' + this.getLineDashOffset() + ';';
      }
      
      return style;
    };
    
    proto.getSimpleCSSFill = function() {
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      return this._stroke ? ( this._stroke.toCSS ? this._stroke.toCSS() : this._stroke ) : 'transparent';
    };
    
    proto.addSVGStrokeDef = function( svg, defs ) {
      var stroke = this.getStroke();
      var strokeId = 'stroke' + this.getId();
      
      // add new definitions if necessary
      if ( stroke && stroke.getSVGDefinition ) {
        defs.appendChild( stroke.getSVGDefinition( strokeId ) );
      }
    };
    
    proto.removeSVGStrokeDef = function( svg, defs ) {
      var strokeId = 'stroke' + this.getId();
      
      // wipe away any old definition
      var oldStrokeDef = svg.getElementById( strokeId );
      if ( oldStrokeDef ) {
        defs.removeChild( oldStrokeDef );
      }
    };
    
    proto.appendStrokablePropString = function( spaces, result ) {
      var self = this;
      
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this._stroke ) {
        var defaultStyles = new LineStyles();
        if ( typeof this._stroke === 'string' ) {
          addProp( 'stroke', this._stroke );
        } else {
          addProp( 'stroke', this._stroke.toString(), true );
        }
        
        _.each( [ 'lineWidth', 'lineCap', 'lineJoin', 'lineDashOffset' ], function( prop ) {
          if ( self[prop] !== defaultStyles[prop] ) {
            addProp( prop, self[prop] );
          }
        } );
        
        if ( this.lineDash.length ) {
          addProp( 'lineDash', JSON.stringify( this.lineDash ), true );
        }
      }
      
      return result;
    };
    
    proto.getStrokeRendererBitmask = function() {
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
    };
    
    // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
    proto._mutatorKeys = [ 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'lineDash', 'lineDashOffset', 'strokePickable' ].concat( proto._mutatorKeys );
    
    // TODO: miterLimit support?
    Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
    Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
    Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
    Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
    Object.defineProperty( proto, 'lineDash', { set: proto.setLineDash, get: proto.getLineDash } );
    Object.defineProperty( proto, 'lineDashOffset', { set: proto.setLineDashOffset, get: proto.getLineDashOffset } );
    Object.defineProperty( proto, 'strokePickable', { set: proto.setStrokePickable, get: proto.isStrokePickable } );
    
    var invalidateStroke = function() {
      this.invalidateSupportedRenderers();
    };
    
    if ( proto.invalidateStroke ) {
      var oldInvalidateStroke = proto.invalidateStroke;
      proto.invalidateStroke = function() {
        this.invalidateSupportedRenderers();
        oldInvalidateStroke.call( this );
      };
    } else {
      proto.invalidateStroke = function() {
        this.invalidateSupportedRenderers();
      };
    }
  };
  var Strokable = scenery.Strokable;
  
  Strokable.StrokableState = function StrokableState( stateType ) {
    var proto = stateType.prototype;
    
    proto.initializeStrokableState = function() {
      this.lastStroke = undefined;
      this.dirtyStroke = true;
      this.dirtyLineWidth = true;
      this.dirtyLineOptions = true; // e.g. cap, join, dash, dashoffset, miterlimit
    };
    
    proto.cleanStrokableState = function() {
      this.dirtyStroke = false;
      this.dirtyLineWidth = false;
      this.dirtyLineOptions = false;
      this.lastStroke = this.node.getStroke();
    };
    
    proto.markDirtyStroke = function() {
      this.dirtyStroke = true;
      this.drawable.markDirty();
    };
    
    proto.markDirtyLineWidth = function() {
      this.dirtyLineWidth = true;
      this.drawable.markDirty();
    };
    
    proto.markDirtyLineOptions = function() {
      this.dirtyLineOptions = true;
      this.drawable.markDirty();
    };
  };
  
  return Strokable;
} );


