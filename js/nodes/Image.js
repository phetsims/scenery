// Copyright 2002-2012, University of Colorado

/**
 * Images
 *
 * TODO: setImage / getImage and the whole toolchain that uses that
 *
 * TODO: SVG support
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' ); // we need to specify the Renderer in the prototype
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  /*
   * Canvas renderer supports the following as 'image':
   *     URL (string)             // works, but does NOT support bounds-based parameter object keys like 'left', 'centerX', etc.
   *                              // also necessary to force updateScene() after it has loaded
   *     HTMLImageElement         // works
   *     HTMLVideoElement         // not tested
   *     HTMLCanvasElement        // works, and forces the canvas renderer
   *     CanvasRenderingContext2D // not tested, but bad luck in past
   *     ImageBitmap              // good luck creating this. currently API for window.createImageBitmap not implemented
   * SVG renderer supports the following as 'image':
   *     URL (string)
   *     HTMLImageElement
   */
  scenery.Image = function Image( image, options ) {
    // allow not passing an options object
    options = options || {};
    
    // rely on the setImage call from the super constructor to do the setup
    if ( image ) {
      options.image = image;
    }
    
    var self = this;
    // allows us to invalidate our bounds whenever an image is loaded
    this.loadListener = function( event ) {
      self.invalidateImage();
      
      // don't leak memory!
      self._image.removeEventListener( 'load', self.loadListener );
    };
    
    Node.call( this, options );
  };
  var Image = scenery.Image;
  
  inherit( Image, Node, {
    invalidateImage: function() {
      this.invalidateSelf( new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
    },
    
    getImage: function() {
      return this._image;
    },
    
    setImage: function( image ) {
      var self = this;
      
      if ( this._image !== image && ( typeof image !== 'string' || !this._image || image !== this._image.src ) ) {
        // don't leak memory by referencing old images
        if ( this._image ) {
          this._image.removeEventListener( 'load', this.loadListener );
        }
        
        if ( typeof image === 'string' ) {
          // create an image with the assumed URL
          var src = image;
          image = document.createElement( 'img' );
          image.addEventListener( 'load', this.loadListener );
          image.src = src;
        } else if ( image instanceof HTMLImageElement ) {
          // only add a listener if we probably haven't loaded yet
          if ( !image.width || !image.height ) {
            image.addEventListener( 'load', this.loadListener );
          }
        }
        
        // swap supported renderers if necessary
        if ( image instanceof HTMLCanvasElement ) {
          if ( !this.hasOwnProperty( '_supportedRenderers' ) ) {
            this._supportedRenderers = [ Renderer.Canvas ];
            this.markLayerRefreshNeeded();
          }
        } else {
          if ( this.hasOwnProperty( '_supportedRenderers' ) ) {
            delete this._supportedRenderers; // will leave prototype intact
            this.markLayerRefreshNeeded();
          }
        }
        
        this._image = image;
        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },
    
    invalidateOnImageLoad: function( image ) {
      var self = this;
      var listener = function( event ) {
        self.invalidateImage();
        
        // don't leak memory!
        image.removeEventListener( listener );
      };
      image.addEventListener( listener );
    },
    
    getImageWidth: function() {
      return this._image.width;
    },
    
    getImageHeight: function() {
      return this._image.height;
    },
    
    getImageURL: function() {
      return this._image.src;
    },
    
    // signal that we are actually rendering something
    isPainted: function() {
      return true;
    },
    
    /*---------------------------------------------------------------------------*
    * Canvas support
    *----------------------------------------------------------------------------*/
    
    // TODO: add SVG / DOM support
    paintCanvas: function( state ) {
      var layer = state.layer;
      var context = layer.context;
      context.drawImage( this._image, 0, 0 );
    },
    
    /*---------------------------------------------------------------------------*
    * WebGL support
    *----------------------------------------------------------------------------*/
    
    paintWebGL: function( state ) {
      throw new Error( 'paintWebGL:nimplemented' );
    },
    
    /*---------------------------------------------------------------------------*
    * SVG support
    *----------------------------------------------------------------------------*/
    
    createSVGFragment: function( svg, defs, group ) {
      var element = document.createElementNS( 'http://www.w3.org/2000/svg', 'image' );
      element.setAttribute( 'x', 0 );
      element.setAttribute( 'y', 0 );
      return element;
    },
    
    updateSVGFragment: function( element ) {
      // like <image xlink:href='http://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
      var xlinkns = 'http://www.w3.org/1999/xlink';
      
      element.setAttribute( 'width', this.getImageWidth() + 'px' );
      element.setAttribute( 'height', this.getImageHeight() + 'px' );
      element.setAttributeNS( xlinkns, 'xlink:href', this.getImageURL() );
    },
    
    /*---------------------------------------------------------------------------*
    * DOM support
    *----------------------------------------------------------------------------*/
    
    getDOMElement: function() {
      this._image.style.display = 'block';
      this._image.style.position = 'absolute';
      return this._image;
    },
    
    updateCSSTransform: function( transform ) {
      $( this._image ).css( transform.getMatrix().getCSSTransformStyles() );
    },
    
    set image( value ) { this.setImage( value ); },
    get image() { return this.getImage(); }
  } );
  
  Image.prototype._mutatorKeys = [ 'image' ].concat( Node.prototype._mutatorKeys );
  
  Image.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG, Renderer.DOM ];
  
  // utility for others
  Image.createSVGImage = function( url, width, height ) {
    var xlinkns = 'http://www.w3.org/1999/xlink';
    var svgns = 'http://www.w3.org/2000/svg';
    
    var element = document.createElementNS( svgns, 'image' );
    element.setAttribute( 'x', 0 );
    element.setAttribute( 'y', 0 );
    element.setAttribute( 'width', width + 'px' );
    element.setAttribute( 'height', height + 'px' );
    element.setAttributeNS( xlinkns, 'xlink:href', url );
    
    return element;
  };
  
  return Image;
} );


