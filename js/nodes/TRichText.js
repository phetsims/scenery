// Copyright 2016-2017, University of Colorado Boulder

/**
 * Wrapper type for scenery phet's RichText node.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  var TNode = require( 'SCENERY/nodes/TNode' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var TString = require( 'ifphetio!PHET_IO/types/TString' );
  var TVoid = require( 'ifphetio!PHET_IO/types/TVoid' );

  /**
   * Wrapper type for scenery's Text node.
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TRichText( text, phetioID ) {
    assert && assertInstanceOf( text, scenery.RichText );
    TNode.call( this, text, phetioID );
  }

  phetioInherit( TNode, 'TRichText', TRichText, {

    setText: {
      returnType: TVoid,
      parameterTypes: [ TString ],
      implementation: function( text ) {
        this.instance.text = text;
      },
      documentation: 'Set the text content'
    },

    getText: {
      returnType: TString,
      parameterTypes: [],
      implementation: function() {
        return this.instance.text;
      },
      documentation: 'Get the text content'
    }
  }, {
    documentation: 'The tandem wrapper type for the scenery RichText node'
  } );

  scenery.register( 'TRichText', TRichText );

  return TRichText;
} );