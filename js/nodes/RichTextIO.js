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
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * Wrapper type for scenery's Text node.
   * @param {Text} richText
   * @param {string} phetioID
   * @constructor
   */
  function RichTextIO( richText, phetioID ) {
    assert && assertInstanceOf( richText, scenery.RichText );
    NodeIO.call( this, richText, phetioID );
  }

  phetioInherit( NodeIO, 'RichTextIO', RichTextIO, {

    setText: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( text ) {
        this.instance.text = text;
      },
      documentation: 'Set the text content'
    },

    getText: {
      returnType: StringIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.text;
      },
      documentation: 'Get the text content'
    }
  }, {
    documentation: 'The tandem wrapper type for the scenery RichText node'
  } );

  scenery.register( 'RichTextIO', RichTextIO );

  return RichTextIO;
} );