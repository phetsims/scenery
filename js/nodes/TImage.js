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
  var TNode = require( 'SCENERY/nodes/TNode' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var TString = require( 'ifphetio!PHET_IO/types/TString' );
  var TVoid = require( 'ifphetio!PHET_IO/types/TVoid' );

  /**
   * Wrapper type for scenery's Text node.
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TImage( text, phetioID ) {
    assert && assertInstanceOf( text, phet.scenery.Image );
    TNode.call( this, text, phetioID );
  }

  phetioInherit( TNode, 'TImage', TImage, {

    setImage: {
      returnType: TVoid,
      parameterTypes: [ TString ],
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

  scenery.register( 'TImage', TImage );

  return TImage;
} );
