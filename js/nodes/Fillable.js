// Copyright 2002-2014, University of Colorado Boulder

/**
 * Mix-in for nodes that support a standard fill.
 *
 * TODO: pattern and gradient handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var platform = require( 'PHET_CORE/platform' );

  var isSafari5 = platform.safari5;

  scenery.Fillable = function Fillable( type ) {
    var proto = type.prototype;

    // this should be called in the constructor to initialize
    proto.initializeFillable = function() {
      this._fill = null;
      this._fillPickable = true;
      this._fillKept = false; // whether the SVG fill should be kept (makes gradients/patterns stay in memory!)

      var that = this;
      this._fillListener = function() {
        that.invalidatePaint(); // TODO: move this to invalidateFill?
        that.invalidateFill();
      };
    };

    proto.hasFill = function() {
      return this._fill !== null;
    };

    proto.getFill = function() {
      return this._fill;
    };

    proto.setFill = function( fill ) {
      if ( this.getFill() !== fill ) {
        var hasInstances = this._instances.length > 0;

        if ( hasInstances && this._fill && this._fill.removeChangeListener ) {
          this._fill.removeChangeListener( this._fillListener );
        }

        this._fill = fill;

        if ( hasInstances && this._fill && this._fill.addChangeListener ) {
          this._fill.addChangeListener( this._fillListener );
        }

        this.invalidatePaint();

        this.invalidateFill();
      }
      return this;
    };

    proto.isFillPickable = function() {
      return this._fillPickable;
    };

    proto.setFillPickable = function( pickable ) {
      assert && assert( typeof pickable === 'boolean' );
      if ( this._fillPickable !== pickable ) {
        this._fillPickable = pickable;

        // TODO: better way of indicating that only the node under pointers could have changed, but no paint change is needed?
        this.invalidateFill();
      }
      return this;
    };

    proto.isFillKept = function() {
      return this._fillKept;
    };

    proto.setFillKept = function( kept ) {
      assert && assert( typeof kept === 'boolean' );
      this._fillKept = kept;
    };

    var superFirstInstanceAdded = proto.firstInstanceAdded;
    proto.firstInstanceAdded = function() {
      if ( this._fill && this._fill.addChangeListener ) {
        this._fill.addChangeListener( this._fillListener );
      }

      if ( superFirstInstanceAdded ) {
        superFirstInstanceAdded.call( this );
      }
    };

    var superLastInstanceRemoved = proto.lastInstanceRemoved;
    proto.lastInstanceRemoved = function() {
      if ( this._fill && this._fill.removeChangeListener ) {
        this._fill.removeChangeListener( this._fillListener );
      }

      if ( superLastInstanceRemoved ) {
        superLastInstanceRemoved.call( this );
      }
    };

    proto.beforeCanvasFill = function( wrapper ) {
      wrapper.setFillStyle( this._fill );
      if ( this._fill.transformMatrix ) {
        wrapper.context.save();
        this._fill.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    };

    proto.afterCanvasFill = function( wrapper ) {
      if ( this._fill.transformMatrix ) {
        wrapper.context.restore();
      }
    };

    proto.getSVGFillStyle = function() {
      var style = 'fill: ';
      if ( !this._fill ) {
        // no fill
        style += 'none;';
      }
      else if ( this._fill.toCSS ) {
        // Color object fill
        style += this._fill.toCSS() + ';';
      }
      else if ( this._fill.getSVGDefinition ) {
        // reference the SVG definition with a URL
        style += 'url(#' + this.getFillId() + ');';
      }
      else {
        // plain CSS color
        style += this._fill + ';';
      }
      return style;
    };

    proto.getFillId = function() {
      return ( this._fill ? this._fill.id : null ) + '-' + this.getId();
    };

    proto.getCSSFill = function() {
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      return this._fill ? ( this._fill.toCSS ? this._fill.toCSS() : this._fill ) : 'transparent';
    };

    proto.addSVGFillDef = function( svg, defs ) {
      var fill = this.getFill();

      // add new definitions if necessary
      if ( fill && fill.getSVGDefinition ) {
        var fillId = this.getFillId();
        defs.appendChild( fill.getSVGDefinition( fillId ) );
      }
    };

    proto.removeSVGFillDef = function( svg, defs ) {
      var fillId = this.getFillId();

      // wipe away any old definition
      var oldFillDef = svg.getElementById( fillId );
      if ( oldFillDef && !this._fillKept ) {
        defs.removeChild( oldFillDef );
      }
    };

    proto.appendFillablePropString = function( spaces, result ) {
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
    };

    proto.getFillRendererBitmask = function() {
      var bitmask = 0;

      // Safari 5 has buggy issues with SVG gradients
      if ( !( isSafari5 && this._fill && this._fill.isGradient ) ) {
        bitmask |= scenery.bitmaskSupportsSVG;
      }

      // we always have Canvas support?
      bitmask |= scenery.bitmaskSupportsCanvas;

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
        // solid fills always supported for DOM and WebGL
        bitmask |= scenery.bitmaskSupportsDOM;
        bitmask |= scenery.bitmaskSupportsWebGL;
      }

      return bitmask;
    };

    // on mutation, set the fill parameter first
    proto._mutatorKeys = [ 'fill', 'fillPickable', 'fillKept' ].concat( proto._mutatorKeys );

    Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
    Object.defineProperty( proto, 'fillPickable', { set: proto.setFillPickable, get: proto.isFillPickable } );
    Object.defineProperty( proto, 'fillKept', { set: proto.setFillKept, get: proto.isFillKept } );

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
  };
  var Fillable = scenery.Fillable;

  return Fillable;
} );


