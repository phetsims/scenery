// Copyright 2013-2016, University of Colorado Boulder


/**
 * Wraps the context and contains a reference to the canvas, so that we can absorb unnecessary state changes,
 * and possibly combine certain fill operations.
 *
 * TODO: performance analysis, possibly axe this and use direct modification.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   *
   * @param {HTMLCanvasElement} canvas
   * @param {CanvasRenderingContext2D} context
   */
  function CanvasContextWrapper( canvas, context ) {

    // @public {HTMLCanvasElement}
    this.canvas = canvas;

    // @public {CanvasRenderingContext2D}
    this.context = context;
    
    this.resetStyles();
  }

  scenery.register( 'CanvasContextWrapper', CanvasContextWrapper );

  inherit( Object, CanvasContextWrapper, {
    // set local styles to undefined, so that they will be invalidated later
    resetStyles: function() {
      this.fillStyle = undefined; // null
      this.strokeStyle = undefined; // null
      this.lineWidth = undefined; // 1
      this.lineCap = undefined; // 'butt'
      this.lineJoin = undefined; // 'miter'
      this.lineDash = undefined; // []
      this.lineDashOffset = undefined; // 0
      this.miterLimit = undefined; // 10

      this.font = undefined; // '10px sans-serif'
      this.direction = undefined; // 'inherit'
    },

    /**
     * Sets a (possibly) new width and height, and clears the canvas.
     * @param width
     * @param height
     */
    setDimensions: function( width, height ) {

      //Don't guard against width and height, because we need to clear the canvas.
      //TODO: Is it expensive to clear by setting both the width and the height?  Maybe we just need to set the width to clear it.
      this.canvas.width = width;
      this.canvas.height = height;

      // assume all persistent data could have changed
      this.resetStyles();
    },

    setFillStyle: function( style ) {
      // turn {Property}s into their values when necessary
      if ( style && style instanceof Property ) {
        style = style.value;
      }

      // turn {Color}s into strings when necessary
      if ( style && style.getCanvasStyle ) {
        style = style.getCanvasStyle();
      }

      if ( this.fillStyle !== style ) {
        this.fillStyle = style;

        // allow gradients / patterns
        this.context.fillStyle = style;
      }
    },

    setStrokeStyle: function( style ) {
      // turn {Property}s into their values when necessary
      if ( style && style instanceof Property ) {
        style = style.value;
      }

      // turn {Color}s into strings when necessary
      if ( style && style.getCanvasStyle ) {
        style = style.getCanvasStyle();
      }

      if ( this.strokeStyle !== style ) {
        this.strokeStyle = style;

        // allow gradients / patterns
        this.context.strokeStyle = style;
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

    setMiterLimit: function( miterLimit ) {
      assert && assert( typeof miterLimit === 'number' );
      if ( this.miterLimit !== miterLimit ) {
        this.miterLimit = miterLimit;
        this.context.miterLimit = miterLimit;
      }
    },

    setLineDash: function( dash ) {
      assert && assert( dash !== undefined, 'undefined line dash would cause hard-to-trace errors' );
      if ( this.lineDash !== dash ) {
        this.lineDash = dash;
        if ( this.context.setLineDash ) {
          this.context.setLineDash( dash === null ? [] : dash ); // see https://github.com/phetsims/scenery/issues/101 for null line-dash workaround
        }
        else if ( this.context.mozDash !== undefined ) {
          this.context.mozDash = dash;
        }
        else if ( this.context.webkitLineDash !== undefined ) {
          this.context.webkitLineDash = dash ? dash : [];
        }
        else {
          // unsupported line dash! do... nothing?
        }
      }
    },

    setLineDashOffset: function( lineDashOffset ) {
      if ( this.lineDashOffset !== lineDashOffset ) {
        this.lineDashOffset = lineDashOffset;
        if ( this.context.lineDashOffset !== undefined ) {
          this.context.lineDashOffset = lineDashOffset;
        }
        else if ( this.context.webkitLineDashOffset !== undefined ) {
          this.context.webkitLineDashOffset = lineDashOffset;
        }
        else {
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
  } );

  return CanvasContextWrapper;
} );
