// Copyright 2016-2017, University of Colorado Boulder

/**
 * IO type for RichText
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );
  var NodeProperty = require( 'SCENERY/util/NodeProperty' );
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );
  var StringIO = require( 'TANDEM/types/StringIO' );

  /**
   * IO type for scenery's Text node.
   * @param {RichText} richText
   * @param {string} phetioID
   * @constructor
   */
  function RichTextIO( richText, phetioID ) {
    NodeIO.call( this, richText, phetioID );

    // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
    var textProperty = new NodeProperty( richText, 'text', 'text', {

      // pick the following values from the parent Node
      phetioReadOnly: richText.phetioReadOnly,
      phetioState: richText.phetioState,
      phetioType: PropertyIO( StringIO ),

      tandem: richText.tandem.createTandem( 'textProperty' ),
      phetioDocumentation: 'Property for the displayed text'
    } );

    // @private
    this.disposeRichTextIO = function() {
      textProperty.dispose();
    };
  }

  phetioInherit( NodeIO, 'RichTextIO', RichTextIO, {

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose: function() {
      this.disposeRichTextIO();
      NodeIO.prototype.dispose.call( this );
    }

  }, {
    documentation: 'The tandem IO type for the scenery RichText node',
    validator: { valueType: scenery.RichText }
  } );

  scenery.register( 'RichTextIO', RichTextIO );

  return RichTextIO;
} );