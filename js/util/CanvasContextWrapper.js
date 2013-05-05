// Copyright 2002-2012, University of Colorado

/**
 * Wraps the context and contains a reference to the canvas, so that we can absorb unnecessary state changes,
 * and possibly combine certain fill operations.
 *
 * TODO: performance analysis, possibly axe this and use direct modification.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.CanvasContextWrapper = function( canvas, context ) {
    this.canvas = canvas;
    this.context = context;
    
    this.resetStyles();
  };
  var CanvasContextWrapper = scenery.CanvasContextWrapper;
  
  CanvasContextWrapper.prototype = {
    constructor: CanvasContextWrapper,
    
    // set local styles to undefined, so that they will be invalidated later
    resetStyles: function() {
      this.fillStyle = undefined; // null
      this.strokeStyle = undefined; // null
      this.lineWidth = undefined; // 1
      this.lineCap = undefined; // 'butt'
      this.lineJoin = undefined; // 'miter'
      this.lineDash = undefined; // null
      this.lineDashOffset = undefined; // 0
      this.miterLimit = undefined; // 10
      
      this.font = undefined; // '10px sans-serif'
      this.direction = undefined; // 'inherit'
    },
    
    setFillStyle: function( style ) {
      if ( this.fillStyle !== style ) {
        this.fillStyle = style;
        
        // allow gradients / patterns
        this.context.fillStyle = ( style && style.getCanvasStyle ) ? style.getCanvasStyle() : style;
      }
    },
    
    setStrokeStyle: function( style ) {
      if ( this.strokeStyle !== style ) {
        this.strokeStyle = style;
        
        // allow gradients / patterns
        this.context.strokeStyle = ( style && style.getCanvasStyle ) ? style.getCanvasStyle() : style;
      }
    },
    
    setLineWidth: function( width ) {
      if ( this.lineWidth !== width ) {
        this.lineWidth = width;
        this.context.lineWidth = width;
      }
    },
    
    setLineCap: function( cap ) {
      if ( this.lineCap !== cap ) {
        this.lineCap = cap;
        this.context.lineCap = cap;
      }
    },
    
    setLineJoin: function( join ) {
      if ( this.lineJoin !== join ) {
        this.lineJoin = join;
        this.context.lineJoin = join;
      }
    },
    
    setLineDash: function( dash ) {
      assert && assert( dash !== undefined, 'undefined line dash would cause hard-to-trace errors' );
      if ( this.lineDash !== dash ) {
        this.lineDash = dash;
        if ( this.context.setLineDash ) {
          this.context.setLineDash( dash );
        } else if ( this.context.mozDash !== undefined ) {
          this.context.mozDash = dash;
        } else if ( this.context.webkitLineDash !== undefined ) {
          this.context.webkitLineDash = dash ? dash : [];
        } else {
          // unsupported line dash! do... nothing?
        }
      }
    },
    
    setLineDashOffset: function( lineDashOffset ) {
      if ( this.lineDashOffset !== lineDashOffset ) {
        this.lineDashOffset = lineDashOffset;
        if ( this.context.lineDashOffset !== undefined ) {
          this.context.lineDashOffset = lineDashOffset;
        } else if ( this.context.webkitLineDashOffset !== undefined ) {
          this.context.webkitLineDashOffset = lineDashOffset;
        } else {
          // unsupported line dash! do... nothing?
        }
      }
    },
    
    setFont: function( font ) {
      if ( this.font !== font ) {
        this.font = font;
        this.context.font = font;
      }
    },
    
    setDirection: function( direction ) {
      if ( this.direction !== direction ) {
        this.direction = direction;
        this.context.direction = direction;
      }
    }
  };
  
  return CanvasContextWrapper;
} );
