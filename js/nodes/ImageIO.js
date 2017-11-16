// Copyright 2016-2017, University of Colorado Boulder

/**
 * Wrapper type for scenery's Image node.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * Wrapper type for scenery's Text node.
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function ImageIO( text, phetioID ) {
    assert && assertInstanceOf( text, phet.scenery.Image );
    NodeIO.call( this, text, phetioID );
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
    documentation: 'The tandem wrapper type for the scenery Text node',
    events: [ 'changed' ]
  } );

  scenery.register( 'ImageIO', ImageIO );

  return ImageIO;
} );
