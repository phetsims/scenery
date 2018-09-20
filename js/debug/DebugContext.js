// Copyright 2013-2016, University of Colorado Boulder

/**
 * A debugging version of the CanvasRenderingContext2D that will output all commands issued,
 * but can also forward them to a real context
 *
 * See the spec at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
 * Wrapping of the CanvasRenderingContext2D interface as of January 27th, 2013 (but not other interfaces like TextMetrics and Path)
 *
 * Shortcut to create:
 *    var context = new scenery.DebugContext( document.createElement( 'canvas' ).getContext( '2d' ) );
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  // used to serialize arguments so that it displays exactly like the call would be executed
  function s( value ) {
    return JSON.stringify( value );
  }

  function log( message ) {
    console.log( 'context.' + message + ';' );
  }

  function attributeGet( name ) {
    log( name );
  }

  function attributeSet( name, value ) {
    log( name + ' = ' + s( value ) );
  }

  function command( name, args ) {
    if ( args === undefined || args.length === 0 ) {
      log( name + '()' );
    }
    else {
      log( name + '( ' + _.reduce( args, function( memo, arg ) {
          if ( memo.length > 0 ) {
            return memo + ', ' + s( arg );
          }
          else {
            return s( arg );
          }
        }, '' ) + ' )' );
    }
  }

  function DebugContext( context ) {
    this._context = context;

    // allow checking of context.ellipse for existence
    if ( context && !context.ellipse ) {
      this.ellipse = context.ellipse;
    }
  }

  scenery.register( 'DebugContext', DebugContext );

  inherit( Object, DebugContext, {
    get canvas() {
      attributeGet( 'canvas' );
      return this._context.canvas;
    },

    get width() {
      attributeGet( 'width' );
      return this._context.width;
    },

    get height() {
      attributeGet( 'height' );
      return this._context.height;
    },

    commit: function() {
      command( 'commit' );
      this._context.commit();
    },

    save: function() {
      command( 'save' );
      this._context.save();
    },

    restore: function() {
      command( 'restore' );
      this._context.restore();
    },

    get currentTransform() {
      attributeGet( 'currentTransform' );
      return this._context.currentTransform;
    },

    set currentTransform( transform ) {
      attributeSet( 'currentTransform', transform );
      this._context.currentTransform = transform;
    },

    scale: function( x, y ) {
      command( 'scale', [ x, y ] );
      this._context.scale( x, y );
    },

    rotate: function( angle ) {
      command( 'rotate', [ angle ] );
      this._context.rotate( angle );
    },

    translate: function( x, y ) {
      command( 'translate', [ x, y ] );
      this._context.translate( x, y );
    },

    transform: function( a, b, c, d, e, f ) {
      command( 'transform', [ a, b, c, d, e, f ] );
      this._context.transform( a, b, c, d, e, f );
    },

    setTransform: function( a, b, c, d, e, f ) {
      command( 'setTransform', [ a, b, c, d, e, f ] );
      this._context.setTransform( a, b, c, d, e, f );
    },

    resetTransform: function() {
      command( 'resetTransform' );
      this._context.resetTransform();
    },

    get globalAlpha() {
      attributeGet( 'globalAlpha' );
      return this._context.globalAlpha;
    },

    set globalAlpha( value ) {
      attributeSet( 'globalAlpha', value );
      this._context.globalAlpha = value;
    },

    get globalCompositeOperation() {
      attributeGet( 'globalCompositeOperation' );
      return this._context.globalCompositeOperation;
    },

    set globalCompositeOperation( value ) {
      attributeSet( 'globalCompositeOperation', value );
      this._context.globalCompositeOperation = value;
    },

    get imageSmoothingEnabled() {
      attributeGet( 'imageSmoothingEnabled' );
      return this._context.imageSmoothingEnabled;
    },

    set imageSmoothingEnabled( value ) {
      attributeSet( 'imageSmoothingEnabled', value );
      this._context.imageSmoothingEnabled = value;
    },

    get strokeStyle() {
      attributeGet( 'strokeStyle' );
      return this._context.strokeStyle;
    },

    set strokeStyle( value ) {
      attributeSet( 'strokeStyle', value );
      this._context.strokeStyle = value;
    },

    get fillStyle() {
      attributeGet( 'fillStyle' );
      return this._context.fillStyle;
    },

    set fillStyle( value ) {
      attributeSet( 'fillStyle', value );
      this._context.fillStyle = value;
    },

    // TODO: create wrapper
    createLinearGradient: function( x0, y0, x1, y1 ) {
      command( 'createLinearGradient', [ x0, y0, x1, y1 ] );
      return this._context.createLinearGradient( x0, y0, x1, y1 );
    },

    // TODO: create wrapper
    createRadialGradient: function( x0, y0, r0, x1, y1, r1 ) {
      command( 'createRadialGradient', [ x0, y0, r0, x1, y1, r1 ] );
      return this._context.createRadialGradient( x0, y0, r0, x1, y1, r1 );
    },

    // TODO: create wrapper
    createPattern: function( image, repetition ) {
      command( 'createPattern', [ image, repetition ] );
      return this._context.createPattern( image, repetition );
    },

    get shadowOffsetX() {
      attributeGet( 'shadowOffsetX' );
      return this._context.shadowOffsetX;
    },

    set shadowOffsetX( value ) {
      attributeSet( 'shadowOffsetX', value );
      this._context.shadowOffsetX = value;
    },

    get shadowOffsetY() {
      attributeGet( 'shadowOffsetY' );
      return this._context.shadowOffsetY;
    },

    set shadowOffsetY( value ) {
      attributeSet( 'shadowOffsetY', value );
      this._context.shadowOffsetY = value;
    },

    get shadowBlur() {
      attributeGet( 'shadowBlur' );
      return this._context.shadowBlur;
    },

    set shadowBlur( value ) {
      attributeSet( 'shadowBlur', value );
      this._context.shadowBlur = value;
    },

    get shadowColor() {
      attributeGet( 'shadowColor' );
      return this._context.shadowColor;
    },

    set shadowColor( value ) {
      attributeSet( 'shadowColor', value );
      this._context.shadowColor = value;
    },

    clearRect: function( x, y, w, h ) {
      command( 'clearRect', [ x, y, w, h ] );
      this._context.clearRect( x, y, w, h );
    },

    fillRect: function( x, y, w, h ) {
      command( 'fillRect', [ x, y, w, h ] );
      this._context.fillRect( x, y, w, h );
    },

    strokeRect: function( x, y, w, h ) {
      command( 'strokeRect', [ x, y, w, h ] );
      this._context.strokeRect( x, y, w, h );
    },

    get fillRule() {
      attributeGet( 'fillRule' );
      return this._context.fillRule;
    },

    set fillRule( value ) {
      attributeSet( 'fillRule', value );
      this._context.fillRule = value;
    },

    beginPath: function() {
      command( 'beginPath' );
      this._context.beginPath();
    },

    fill: function( path ) {
      if ( path ) {
        command( 'fill', [ path ] );
        this._context.fill( path );
      }
      else {
        command( 'fill' );
        this._context.fill();
      }
    },

    stroke: function( path ) {
      if ( path ) {
        command( 'stroke', [ path ] );
        this._context.stroke( path );
      }
      else {
        command( 'stroke' );
        this._context.stroke();
      }
    },

    drawSystemFocusRing: function( a, b ) {
      command( 'drawSystemFocusRing', b ? [ a, b ] : [ a ] );
      this._context.drawSystemFocusRing( a, b );
    },

    drawCustomFocusRing: function( a, b ) {
      command( 'drawCustomFocusRing', b ? [ a, b ] : [ a ] );
      return this._context.drawCustomFocusRing( a, b );
    },

    scrollPathIntoView: function( path ) {
      command( 'scrollPathIntoView', path ? [ path ] : undefined );
      this._context.scrollPathIntoView( path );
    },

    clip: function( path ) {
      command( 'clip', path ? [ path ] : undefined );
      this._context.clip( path );
    },

    resetClip: function() {
      command( 'resetClip' );
      this._context.resetClip();
    },

    isPointInPath: function( a, b, c ) {
      command( 'isPointInPath', c ? [ a, b, c ] : [ a, b ] );
      return this._context.isPointInPath( a, b, c );
    },

    fillText: function( text, x, y, maxWidth ) {
      command( 'fillText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
      this._context.fillText( text, x, y, maxWidth );
    },

    strokeText: function( text, x, y, maxWidth ) {
      command( 'strokeText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
      this._context.strokeText( text, x, y, maxWidth );
    },

    measureText: function( text ) {
      command( 'measureText', [ text ] );
      return this._context.measureText( text );
    },

    drawImage: function( image, a, b, c, d, e, f, g, h ) {
      command( 'drawImage', c !== undefined ? ( e !== undefined ? [ image, a, b, c, d, e, f, g, h ] : [ image, a, b, c, d ] ) : [ image, a, b ] );
      this._context.drawImage( image, a, b, c, d, e, f, g, h );
    },

    addHitRegion: function( options ) {
      command( 'addHitRegion', [ options ] );
      this._context.addHitRegion( options );
    },

    removeHitRegion: function( options ) {
      command( 'removeHitRegion', [ options ] );
      this._context.removeHitRegion( options );
    },

    createImageData: function( a, b ) {
      command( 'createImageData', b !== undefined ? [ a, b ] : [ a ] );
      return this._context.createImageData( a, b );
    },

    createImageDataHD: function( a, b ) {
      command( 'createImageDataHD', [ a, b ] );
      return this._context.createImageDataHD( a, b );
    },

    getImageData: function( sx, sy, sw, sh ) {
      command( 'getImageData', [ sx, sy, sw, sh ] );
      return this._context.getImageData( sx, sy, sw, sh );
    },

    getImageDataHD: function( sx, sy, sw, sh ) {
      command( 'getImageDataHD', [ sx, sy, sw, sh ] );
      return this._context.getImageDataHD( sx, sy, sw, sh );
    },

    putImageData: function( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
      command( 'putImageData', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
      this._context.putImageData( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
    },

    putImageDataHD: function( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
      command( 'putImageDataHD', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
      this._context.putImageDataHD( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
    },

    /*---------------------------------------------------------------------------*
     * CanvasDrawingStyles
     *----------------------------------------------------------------------------*/

    get lineWidth() {
      attributeGet( 'lineWidth' );
      return this._context.lineWidth;
    },

    set lineWidth( value ) {
      attributeSet( 'lineWidth', value );
      this._context.lineWidth = value;
    },

    get lineCap() {
      attributeGet( 'lineCap' );
      return this._context.lineCap;
    },

    set lineCap( value ) {
      attributeSet( 'lineCap', value );
      this._context.lineCap = value;
    },

    get lineJoin() {
      attributeGet( 'lineJoin' );
      return this._context.lineJoin;
    },

    set lineJoin( value ) {
      attributeSet( 'lineJoin', value );
      this._context.lineJoin = value;
    },

    get miterLimit() {
      attributeGet( 'miterLimit' );
      return this._context.miterLimit;
    },

    set miterLimit( value ) {
      attributeSet( 'miterLimit', value );
      this._context.miterLimit = value;
    },

    setLineDash: function( segments ) {
      command( 'setLineDash', [ segments ] );
      this._context.setLineDash( segments );
    },

    getLineDash: function() {
      command( 'getLineDash' );
      return this._context.getLineDash();
    },

    get lineDashOffset() {
      attributeGet( 'lineDashOffset' );
      return this._context.lineDashOffset;
    },

    set lineDashOffset( value ) {
      attributeSet( 'lineDashOffset', value );
      this._context.lineDashOffset = value;
    },

    get font() {
      attributeGet( 'font' );
      return this._context.font;
    },

    set font( value ) {
      attributeSet( 'font', value );
      this._context.font = value;
    },

    get textAlign() {
      attributeGet( 'textAlign' );
      return this._context.textAlign;
    },

    set textAlign( value ) {
      attributeSet( 'textAlign', value );
      this._context.textAlign = value;
    },

    get textBaseline() {
      attributeGet( 'textBaseline' );
      return this._context.textBaseline;
    },

    set textBaseline( value ) {
      attributeSet( 'textBaseline', value );
      this._context.textBaseline = value;
    },

    get direction() {
      attributeGet( 'direction' );
      return this._context.direction;
    },

    set direction( value ) {
      attributeSet( 'direction', value );
      this._context.direction = value;
    },

    /*---------------------------------------------------------------------------*
     * CanvasPathMethods
     *----------------------------------------------------------------------------*/

    closePath: function() {
      command( 'closePath' );
      this._context.closePath();
    },

    moveTo: function( x, y ) {
      command( 'moveTo', [ x, y ] );
      this._context.moveTo( x, y );
    },

    lineTo: function( x, y ) {
      command( 'lineTo', [ x, y ] );
      this._context.lineTo( x, y );
    },

    quadraticCurveTo: function( cpx, cpy, x, y ) {
      command( 'quadraticCurveTo', [ cpx, cpy, x, y ] );
      this._context.quadraticCurveTo( cpx, cpy, x, y );
    },

    bezierCurveTo: function( cp1x, cp1y, cp2x, cp2y, x, y ) {
      command( 'bezierCurveTo', [ cp1x, cp1y, cp2x, cp2y, x, y ] );
      this._context.bezierCurveTo( cp1x, cp1y, cp2x, cp2y, x, y );
    },

    arcTo: function( x1, y1, x2, y2, radiusX, radiusY, rotation ) {
      command( 'arcTo', radiusY !== undefined ? [ x1, y1, x2, y2, radiusX, radiusY, rotation ] : [ x1, y1, x2, y2, radiusX ] );
      this._context.arcTo( x1, y1, x2, y2, radiusX, radiusY, rotation );
    },

    rect: function( x, y, w, h ) {
      command( 'rect', [ x, y, w, h ] );
      this._context.rect( x, y, w, h );
    },

    arc: function( x, y, radius, startAngle, endAngle, anticlockwise ) {
      command( 'arc', [ x, y, radius, startAngle, endAngle, anticlockwise ] );
      this._context.arc( x, y, radius, startAngle, endAngle, anticlockwise );
    },

    ellipse: function( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
      command( 'ellipse', [ x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ] );
      this._context.ellipse( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise );
    }
  } );

  return DebugContext;
} );


