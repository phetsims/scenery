// Copyright 2013-2023, University of Colorado Boulder

/**
 * A debugging version of the CanvasRenderingContext2D that will output all commands issued,
 * but can also forward them to a real context
 *
 * See the spec at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
 * Wrapping of the CanvasRenderingContext2D interface as of January 27th, 2013 (but not other interfaces like TextMetrics and Path)
 *
 * Shortcut to create:
 *    var context = new phet.scenery.DebugContext( document.createElement( 'canvas' ).getContext( '2d' ) );
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

// used to serialize arguments so that it displays exactly like the call would be executed
function s( value ) {
  return JSON.stringify( value );
}

function log( message ) {
  console.log( `context.${message};` );
}

function attributeGet( name ) {
  log( name );
}

function attributeSet( name, value ) {
  log( `${name} = ${s( value )}` );
}

function command( name, args ) {
  if ( args === undefined || args.length === 0 ) {
    log( `${name}()` );
  }
  else {
    log( `${name}( ${_.reduce( args, ( memo, arg ) => {
      if ( memo.length > 0 ) {
        return `${memo}, ${s( arg )}`;
      }
      else {
        return s( arg );
      }
    }, '' )} )` );
  }
}

class DebugContext {
  /**
   * @param {CanvasRenderingContext2D} context
   */
  constructor( context ) {
    this._context = context;

    // allow checking of context.ellipse for existence
    if ( context && !context.ellipse ) {
      this.ellipse = context.ellipse;
    }
  }

  /**
   * @public
   *
   * @returns {HTMLCanvasElement}
   */
  get canvas() {
    attributeGet( 'canvas' );
    return this._context.canvas;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get width() {
    attributeGet( 'width' );
    return this._context.width;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get height() {
    attributeGet( 'height' );
    return this._context.height;
  }

  /**
   * @public
   */
  commit() {
    command( 'commit' );
    this._context.commit();
  }

  /**
   * @public
   */
  save() {
    command( 'save' );
    this._context.save();
  }

  /**
   * @public
   */
  restore() {
    command( 'restore' );
    this._context.restore();
  }

  /**
   * @public
   *
   * @returns {DOMMatrix}
   */
  get currentTransform() {
    attributeGet( 'currentTransform' );
    return this._context.currentTransform;
  }

  /**
   * @public
   *
   * @param {DOMMatrix} transform
   */
  set currentTransform( transform ) {
    attributeSet( 'currentTransform', transform );
    this._context.currentTransform = transform;
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   */
  scale( x, y ) {
    command( 'scale', [ x, y ] );
    this._context.scale( x, y );
  }

  /**
   * @public
   *
   * @param {number} angle
   */
  rotate( angle ) {
    command( 'rotate', [ angle ] );
    this._context.rotate( angle );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   */
  translate( x, y ) {
    command( 'translate', [ x, y ] );
    this._context.translate( x, y );
  }

  /**
   * @public
   *
   * @param {number} a
   * @param {number} b
   * @param {number} c
   * @param {number} d
   * @param {number} e
   * @param {number} f
   */
  transform( a, b, c, d, e, f ) {
    command( 'transform', [ a, b, c, d, e, f ] );
    this._context.transform( a, b, c, d, e, f );
  }

  /**
   * @public
   *
   * @param {number} a
   * @param {number} b
   * @param {number} c
   * @param {number} d
   * @param {number} e
   * @param {number} f
   */
  setTransform( a, b, c, d, e, f ) {
    command( 'setTransform', [ a, b, c, d, e, f ] );
    this._context.setTransform( a, b, c, d, e, f );
  }

  /**
   * @public
   */
  resetTransform() {
    command( 'resetTransform' );
    this._context.resetTransform();
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get globalAlpha() {
    attributeGet( 'globalAlpha' );
    return this._context.globalAlpha;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set globalAlpha( value ) {
    attributeSet( 'globalAlpha', value );
    this._context.globalAlpha = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get globalCompositeOperation() {
    attributeGet( 'globalCompositeOperation' );
    return this._context.globalCompositeOperation;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set globalCompositeOperation( value ) {
    attributeSet( 'globalCompositeOperation', value );
    this._context.globalCompositeOperation = value;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get imageSmoothingEnabled() {
    attributeGet( 'imageSmoothingEnabled' );
    return this._context.imageSmoothingEnabled;
  }

  /**
   * @public
   *
   * @param {boolean} value
   */
  set imageSmoothingEnabled( value ) {
    attributeSet( 'imageSmoothingEnabled', value );
    this._context.imageSmoothingEnabled = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get strokeStyle() {
    attributeGet( 'strokeStyle' );
    return this._context.strokeStyle;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set strokeStyle( value ) {
    attributeSet( 'strokeStyle', value );
    this._context.strokeStyle = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get fillStyle() {
    attributeGet( 'fillStyle' );
    return this._context.fillStyle;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set fillStyle( value ) {
    attributeSet( 'fillStyle', value );
    this._context.fillStyle = value;
  }

  // TODO: create wrapper
  /**
   * @public
   *
   * @param {number} x0
   * @param {number} y0
   * @param {number} x1
   * @param {number} y1
   * @returns {*}
   */
  createLinearGradient( x0, y0, x1, y1 ) {
    command( 'createLinearGradient', [ x0, y0, x1, y1 ] );
    return this._context.createLinearGradient( x0, y0, x1, y1 );
  }

  // TODO: create wrapper
  /**
   * @public
   *
   * @param {number} x0
   * @param {number} y0
   * @param {number} r0
   * @param {number} x1
   * @param {number} y1
   * @param {number} r1
   * @returns {*}
   */
  createRadialGradient( x0, y0, r0, x1, y1, r1 ) {
    command( 'createRadialGradient', [ x0, y0, r0, x1, y1, r1 ] );
    return this._context.createRadialGradient( x0, y0, r0, x1, y1, r1 );
  }

  // TODO: create wrapper
  /**
   * @public
   *
   * @param {*} image
   * @param {string} repetition
   * @returns {*}
   */
  createPattern( image, repetition ) {
    command( 'createPattern', [ image, repetition ] );
    return this._context.createPattern( image, repetition );
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get shadowOffsetX() {
    attributeGet( 'shadowOffsetX' );
    return this._context.shadowOffsetX;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set shadowOffsetX( value ) {
    attributeSet( 'shadowOffsetX', value );
    this._context.shadowOffsetX = value;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get shadowOffsetY() {
    attributeGet( 'shadowOffsetY' );
    return this._context.shadowOffsetY;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set shadowOffsetY( value ) {
    attributeSet( 'shadowOffsetY', value );
    this._context.shadowOffsetY = value;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get shadowBlur() {
    attributeGet( 'shadowBlur' );
    return this._context.shadowBlur;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set shadowBlur( value ) {
    attributeSet( 'shadowBlur', value );
    this._context.shadowBlur = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get shadowColor() {
    attributeGet( 'shadowColor' );
    return this._context.shadowColor;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set shadowColor( value ) {
    attributeSet( 'shadowColor', value );
    this._context.shadowColor = value;
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   */
  clearRect( x, y, w, h ) {
    command( 'clearRect', [ x, y, w, h ] );
    this._context.clearRect( x, y, w, h );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   */
  fillRect( x, y, w, h ) {
    command( 'fillRect', [ x, y, w, h ] );
    this._context.fillRect( x, y, w, h );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   */
  strokeRect( x, y, w, h ) {
    command( 'strokeRect', [ x, y, w, h ] );
    this._context.strokeRect( x, y, w, h );
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get fillRule() {
    attributeGet( 'fillRule' );
    return this._context.fillRule;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set fillRule( value ) {
    attributeSet( 'fillRule', value );
    this._context.fillRule = value;
  }

  /**
   * @public
   */
  beginPath() {
    command( 'beginPath' );
    this._context.beginPath();
  }

  /**
   * @public
   *
   * @param {Path2D} path
   */
  fill( path ) {
    if ( path ) {
      command( 'fill', [ path ] );
      this._context.fill( path );
    }
    else {
      command( 'fill' );
      this._context.fill();
    }
  }

  /**
   * @public
   *
   * @param {Path2D} path
   */
  stroke( path ) {
    if ( path ) {
      command( 'stroke', [ path ] );
      this._context.stroke( path );
    }
    else {
      command( 'stroke' );
      this._context.stroke();
    }
  }

  /**
   * @public
   *
   * @param {Path2D} path
   */
  scrollPathIntoView( path ) {
    command( 'scrollPathIntoView', path ? [ path ] : undefined );
    this._context.scrollPathIntoView( path );
  }

  /**
   * @public
   *
   * @param {Path2D} path
   */
  clip( path ) {
    command( 'clip', path ? [ path ] : undefined );
    this._context.clip( path );
  }

  /**
   * @public
   */
  resetClip() {
    command( 'resetClip' );
    this._context.resetClip();
  }

  /**
   * @public
   *
   * @param {*} a
   * @param {*} b
   * @param {*} c
   * @returns {*}
   */
  isPointInPath( a, b, c ) {
    command( 'isPointInPath', c ? [ a, b, c ] : [ a, b ] );
    return this._context.isPointInPath( a, b, c );
  }

  /**
   * @public
   *
   * @param {*} text
   * @param {number} x
   * @param {number} y
   * @param {*} maxWidth
   */
  fillText( text, x, y, maxWidth ) {
    command( 'fillText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
    this._context.fillText( text, x, y, maxWidth );
  }

  /**
   * @public
   *
   * @param {*} text
   * @param {number} x
   * @param {number} y
   * @param {*} maxWidth
   */
  strokeText( text, x, y, maxWidth ) {
    command( 'strokeText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
    this._context.strokeText( text, x, y, maxWidth );
  }

  /**
   * @public
   *
   * @param {*} text
   * @returns {*}
   */
  measureText( text ) {
    command( 'measureText', [ text ] );
    return this._context.measureText( text );
  }

  /**
   * @public
   *
   * @param {*} image
   * @param {*} a
   * @param {*} b
   * @param {*} c
   * @param {*} d
   * @param {*} e
   * @param {*} f
   * @param {*} g
   * @param {number} h
   */
  drawImage( image, a, b, c, d, e, f, g, h ) {
    command( 'drawImage', c !== undefined ? ( e !== undefined ? [ image, a, b, c, d, e, f, g, h ] : [ image, a, b, c, d ] ) : [ image, a, b ] );
    this._context.drawImage( image, a, b, c, d, e, f, g, h );
  }

  /**
   * @public
   *
   * @param {[Object]} options
   */
  addHitRegion( options ) {
    command( 'addHitRegion', [ options ] );
    this._context.addHitRegion( options );
  }

  /**
   * @public
   *
   * @param {[Object]} options
   */
  removeHitRegion( options ) {
    command( 'removeHitRegion', [ options ] );
    this._context.removeHitRegion( options );
  }

  /**
   * @public
   *
   * @param {*} a
   * @param {*} b
   * @returns {*}
   */
  createImageData( a, b ) {
    command( 'createImageData', b !== undefined ? [ a, b ] : [ a ] );
    return this._context.createImageData( a, b );
  }

  /**
   * @public
   *
   * @param {*} a
   * @param {*} b
   * @returns {*}
   */
  createImageDataHD( a, b ) {
    command( 'createImageDataHD', [ a, b ] );
    return this._context.createImageDataHD( a, b );
  }

  /**
   * @public
   *
   * @param {*} sx
   * @param {*} sy
   * @param {*} sw
   * @param {*} sh
   * @returns {*}
   */
  getImageData( sx, sy, sw, sh ) {
    command( 'getImageData', [ sx, sy, sw, sh ] );
    return this._context.getImageData( sx, sy, sw, sh );
  }

  /**
   * @public
   *
   * @param {*} sx
   * @param {*} sy
   * @param {*} sw
   * @param {*} sh
   * @returns {*}
   */
  getImageDataHD( sx, sy, sw, sh ) {
    command( 'getImageDataHD', [ sx, sy, sw, sh ] );
    return this._context.getImageDataHD( sx, sy, sw, sh );
  }

  /**
   * @public
   *
   * @param {*} imageData
   * @param {*} dx
   * @param {*} dy
   * @param {*} dirtyX
   * @param {*} dirtyY
   * @param {*} dirtyWidth
   * @param {*} dirtyHeight
   */
  putImageData( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
    command( 'putImageData', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
    this._context.putImageData( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
  }

  /**
   * @public
   *
   * @param {*} imageData
   * @param {*} dx
   * @param {*} dy
   * @param {*} dirtyX
   * @param {*} dirtyY
   * @param {*} dirtyWidth
   * @param {*} dirtyHeight
   */
  putImageDataHD( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
    command( 'putImageDataHD', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
    this._context.putImageDataHD( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
  }

  /*---------------------------------------------------------------------------*
   * CanvasDrawingStyles
   *----------------------------------------------------------------------------*/

  /**
   * @public
   *
   * @returns {number}
   */
  get lineWidth() {
    attributeGet( 'lineWidth' );
    return this._context.lineWidth;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set lineWidth( value ) {
    attributeSet( 'lineWidth', value );
    this._context.lineWidth = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get lineCap() {
    attributeGet( 'lineCap' );
    return this._context.lineCap;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set lineCap( value ) {
    attributeSet( 'lineCap', value );
    this._context.lineCap = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get lineJoin() {
    attributeGet( 'lineJoin' );
    return this._context.lineJoin;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set lineJoin( value ) {
    attributeSet( 'lineJoin', value );
    this._context.lineJoin = value;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get miterLimit() {
    attributeGet( 'miterLimit' );
    return this._context.miterLimit;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set miterLimit( value ) {
    attributeSet( 'miterLimit', value );
    this._context.miterLimit = value;
  }

  /**
   * @public
   *
   * @param {*} segments
   */
  setLineDash( segments ) {
    command( 'setLineDash', [ segments ] );
    this._context.setLineDash( segments );
  }

  /**
   * @public
   * @returns {*}
   */
  getLineDash() {
    command( 'getLineDash' );
    return this._context.getLineDash();
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get lineDashOffset() {
    attributeGet( 'lineDashOffset' );
    return this._context.lineDashOffset;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set lineDashOffset( value ) {
    attributeSet( 'lineDashOffset', value );
    this._context.lineDashOffset = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get font() {
    attributeGet( 'font' );
    return this._context.font;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set font( value ) {
    attributeSet( 'font', value );
    this._context.font = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get textAlign() {
    attributeGet( 'textAlign' );
    return this._context.textAlign;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set textAlign( value ) {
    attributeSet( 'textAlign', value );
    this._context.textAlign = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get textBaseline() {
    attributeGet( 'textBaseline' );
    return this._context.textBaseline;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set textBaseline( value ) {
    attributeSet( 'textBaseline', value );
    this._context.textBaseline = value;
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get direction() {
    attributeGet( 'direction' );
    return this._context.direction;
  }

  /**
   * @public
   *
   * @param {string} value
   */
  set direction( value ) {
    attributeSet( 'direction', value );
    this._context.direction = value;
  }

  /*---------------------------------------------------------------------------*
   * CanvasPathMethods
   *----------------------------------------------------------------------------*/

  /**
   * @public
   */
  closePath() {
    command( 'closePath' );
    this._context.closePath();
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   */
  moveTo( x, y ) {
    command( 'moveTo', [ x, y ] );
    this._context.moveTo( x, y );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   */
  lineTo( x, y ) {
    command( 'lineTo', [ x, y ] );
    this._context.lineTo( x, y );
  }

  /**
   * @public
   *
   * @param {*} cpx
   * @param {*} cpy
   * @param {number} x
   * @param {number} y
   */
  quadraticCurveTo( cpx, cpy, x, y ) {
    command( 'quadraticCurveTo', [ cpx, cpy, x, y ] );
    this._context.quadraticCurveTo( cpx, cpy, x, y );
  }

  /**
   * @public
   *
   * @param {*} cp1x
   * @param {*} cp1y
   * @param {*} cp2x
   * @param {*} cp2y
   * @param {number} x
   * @param {number} y
   */
  bezierCurveTo( cp1x, cp1y, cp2x, cp2y, x, y ) {
    command( 'bezierCurveTo', [ cp1x, cp1y, cp2x, cp2y, x, y ] );
    this._context.bezierCurveTo( cp1x, cp1y, cp2x, cp2y, x, y );
  }

  /**
   * @public
   *
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {number} radiusX
   * @param {number} radiusY
   * @param {number} rotation
   */
  arcTo( x1, y1, x2, y2, radiusX, radiusY, rotation ) {
    command( 'arcTo', radiusY !== undefined ? [ x1, y1, x2, y2, radiusX, radiusY, rotation ] : [ x1, y1, x2, y2, radiusX ] );
    this._context.arcTo( x1, y1, x2, y2, radiusX, radiusY, rotation );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   */
  rect( x, y, w, h ) {
    command( 'rect', [ x, y, w, h ] );
    this._context.rect( x, y, w, h );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} radius
   * @param {number} startAngle
   * @param {number} endAngle
   * @param {boolean} anticlockwise
   */
  arc( x, y, radius, startAngle, endAngle, anticlockwise ) {
    command( 'arc', [ x, y, radius, startAngle, endAngle, anticlockwise ] );
    this._context.arc( x, y, radius, startAngle, endAngle, anticlockwise );
  }

  /**
   * @public
   *
   * @param {number} x
   * @param {number} y
   * @param {number} radiusX
   * @param {number} radiusY
   * @param {number} rotation
   * @param {number} startAngle
   * @param {number} endAngle
   * @param {boolean} anticlockwise
   */
  ellipse( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
    command( 'ellipse', [ x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ] );
    this._context.ellipse( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise );
  }
}

scenery.register( 'DebugContext', DebugContext );
export default DebugContext;