// Copyright 2002-2013, University of Colorado

/**
 * Images
 *
 * TODO: allow multiple DOM instances (create new HTMLImageElement elements)
 * TODO: SVG support
 * TODO: support rendering a Canvas to DOM (single instance)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' ); // we need to specify the Renderer in the prototype
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  require( 'SCENERY/util/Util' );

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
    assert && assert( image, "image should be available" );

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
    this.invalidateSupportedRenderers();
  };
  var Image = scenery.Image;

  inherit( Node, Image, {
    allowsMultipleDOMInstances: false, // TODO: support multiple instances

    invalidateImage: function() {
      this.invalidateSelf( new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
    },

    getImage: function() {
      return this._image;
    },
    
    invalidateSupportedRenderers: function() {
      if ( this._image instanceof HTMLCanvasElement ) {
        this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas );
      } else {
        // assumes HTMLImageElement
        this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskSupportsDOM );
      }
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
        this.invalidateSupportedRenderers();

        this._image = image;
        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },

    getImageWidth: function() {
      return this._image.naturalWidth || this._image.width;
    },

    getImageHeight: function() {
      return this._image.naturalHeight || this._image.height;
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
    paintCanvas: function( wrapper ) {
      wrapper.context.drawImage( this._image, 0, 0 );
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
      var element = document.createElementNS( scenery.svgns, 'image' );
      element.setAttribute( 'x', 0 );
      element.setAttribute( 'y', 0 );
      return element;
    },

    updateSVGFragment: function( element ) {
      // like <image xlink:href='http://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
      element.setAttribute( 'width', this.getImageWidth() + 'px' );
      element.setAttribute( 'height', this.getImageHeight() + 'px' );
      element.setAttributeNS( scenery.xlinkns, 'xlink:href', this.getImageURL() );
    },

    /*---------------------------------------------------------------------------*
     * DOM support
     *----------------------------------------------------------------------------*/

    getDOMElement: function() {
      this._image.style.display = 'block';
      this._image.style.position = 'absolute';
      this._image.style.left = '0';
      this._image.style.top = '0';
      return this._image;
    },

    updateDOMElement: function( image ) {
      if ( image.src !== this._image.src ) {
        image.src = this._image.src;
      }
    },

    updateCSSTransform: function( transform, element ) {
      // TODO: extract this out, it's completely shared!
      scenery.Util.applyCSSTransform( transform.getMatrix(), element );
    },

    set image( value ) { this.setImage( value ); },
    get image() { return this.getImage(); },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + ( this._image.src ? this._image.src.replace( /'/g, '\\\'' ) : 'other' ) + '\', {' + propLines + '} )';
    }
  } );

  Image.prototype._mutatorKeys = [ 'image' ].concat( Node.prototype._mutatorKeys );

  // utility for others
  Image.createSVGImage = function( url, width, height ) {
    var element = document.createElementNS( scenery.svgns, 'image' );
    element.setAttribute( 'x', 0 );
    element.setAttribute( 'y', 0 );
    element.setAttribute( 'width', width + 'px' );
    element.setAttribute( 'height', height + 'px' );
    element.setAttributeNS( scenery.xlinkns, 'xlink:href', url );

    return element;
  };

  return Image;
} );


