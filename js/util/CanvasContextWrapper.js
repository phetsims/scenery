// Copyright 2013-2022, University of Colorado Boulder


/**
 * Wraps the context and contains a reference to the canvas, so that we can absorb unnecessary state changes,
 * and possibly combine certain fill operations.
 *
 * TODO: performance analysis, possibly axe this and use direct modification.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import { scenery } from '../imports.js';

class CanvasContextWrapper {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {CanvasRenderingContext2D} context
   */
  constructor( canvas, context ) {

    // @public {HTMLCanvasElement}
    this.canvas = canvas;

    // @public {CanvasRenderingContext2D}
    this.context = context;

    this.resetStyles();
  }

  /**
   * Set local styles to undefined, so that they will be invalidated later
   * @public
   */
  resetStyles() {
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
  }

  /**
   * Sets a (possibly) new width and height, and clears the canvas.
   * @public
   *
   * @param {number} width
   * @param {number} height
   */
  setDimensions( width, height ) {

    //Don't guard against width and height, because we need to clear the canvas.
    //TODO: Is it expensive to clear by setting both the width and the height?  Maybe we just need to set the width to clear it.
    this.canvas.width = width;
    this.canvas.height = height;

    // assume all persistent data could have changed
    this.resetStyles();
  }

  /**
   * @public
   *
   * @param {string|Color|Property.<string>} style
   */
  setFillStyle( style ) {
    // turn {Property}s into their values when necessary
    if ( style && style instanceof ReadOnlyProperty ) {
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
  }

  /**
   * @public
   *
   * @param {string|Color|Property.<string>} style
   */
  setStrokeStyle( style ) {
    // turn {Property}s into their values when necessary
    if ( style && style instanceof ReadOnlyProperty ) {
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
  }

  /**
   * @public
   *
   * @param {number} width
   */
  setLineWidth( width ) {
    if ( this.lineWidth !== width ) {
      this.lineWidth = width;
      this.context.lineWidth = width;
    }
  }

  /**
   * @public
   *
   * @param {string} cap
   */
  setLineCap( cap ) {
    if ( this.lineCap !== cap ) {
      this.lineCap = cap;
      this.context.lineCap = cap;
    }
  }

  /**
   * @public
   *
   * @param {string} join
   */
  setLineJoin( join ) {
    if ( this.lineJoin !== join ) {
      this.lineJoin = join;
      this.context.lineJoin = join;
    }
  }

  /**
   * @public
   *
   * @param {number} miterLimit
   */
  setMiterLimit( miterLimit ) {
    assert && assert( typeof miterLimit === 'number' );
    if ( this.miterLimit !== miterLimit ) {
      this.miterLimit = miterLimit;
      this.context.miterLimit = miterLimit;
    }
  }

  /**
   * @public
   *
   * @param {Array.<number>|null} dash
   */
  setLineDash( dash ) {
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
  }

  /**
   * @public
   *
   * @param {number} lineDashOffset
   */
  setLineDashOffset( lineDashOffset ) {
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
  }

  /**
   * @public
   *
   * @param {string} font
   */
  setFont( font ) {
    if ( this.font !== font ) {
      this.font = font;
      this.context.font = font;
    }
  }

  /**
   * @public
   *
   * @param {string} direction
   */
  setDirection( direction ) {
    if ( this.direction !== direction ) {
      this.direction = direction;
      this.context.direction = direction;
    }
  }
}

scenery.register( 'CanvasContextWrapper', CanvasContextWrapper );
export default CanvasContextWrapper;