// Copyright 2016-2017, University of Colorado Boulder

/**
 * IO type for SCENERY/Image (not the Image HTMLElement)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * @param {Image} image
   * @param {string} phetioID
   * @constructor
   */
  function ImageIO( image, phetioID ) {
    assert && assertInstanceOf( image, phet.scenery.Image );
    NodeIO.call( this, image, phetioID );
  }

  phetioInherit( NodeIO, 'ImageIO', ImageIO, {

    setImage: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( base64Text ) {
        var im = new window.Image();
        im.src = base64Text;
        this.instance.image = im;
      },
      documentation: 'Set the image from a base64 string'
    }
  }, {
    documentation: 'The tandem IO type for the scenery Text node',
    events: [ 'changed' ]
  } );

  scenery.register( 'ImageIO', ImageIO );

  return ImageIO;
} );
