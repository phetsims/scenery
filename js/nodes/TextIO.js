// Copyright 2016-2017, University of Colorado Boulder

/**
 * IO type for Text
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var FontIO = require( 'SCENERY/util/FontIO' );
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );
  var NodeProperty = require( 'SCENERY/util/NodeProperty' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var NumberIO = require( 'ifphetio!PHET_IO/types/NumberIO' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TextIO( text, phetioID ) {
    assert && assertInstanceOf( text, phet.scenery.Text );
    NodeIO.call( this, text, phetioID );

    // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
    var textProperty = new NodeProperty( text, 'text', 'text', {

      // pick the following values from the parent Node
      phetioReadOnly: text.phetioReadOnly,
      phetioState: text.phetioState,
      phetioType: PropertyIO( StringIO ),

      tandem: text.tandem.createTandem( 'textProperty' ),
      phetioInstanceDocumentation: 'Property for the displayed text.'
    } );

    // @private
    this.disposeTextIO = function() {
      textProperty.dispose();
    };
  }

  phetioInherit( NodeIO, 'TextIO', TextIO, {

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose: function() {
      this.disposeTextIO();
    },

    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ FontIO ],
      implementation: function( font ) {
        this.instance.setFont( font );
      },
      documentation: 'Sets font options for this TextIO instance, e.g. {size: 16, weight: bold}. If increasing the font ' +
                     'size does not make the text size larger, you may need to increase the maxWidth of the TextIO also.'
    },

    getFontOptions: {
      returnType: FontIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.getFont();
      },
      documentation: 'Gets font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( maxWidth ) {
        this.instance.setMaxWidth( maxWidth );
      },
      documentation: 'Sets the maximum width of text box. ' +
                     'If the text width exceeds maxWidth, it is scaled down to fit.'
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.maxWidth;
      },
      documentation: 'Gets the maximum width of text box'
    }
  }, {
    documentation: 'Text that is displayed in the simulation.',
    events: [ 'textChanged' ]
  } );

  scenery.register( 'TextIO', TextIO );

  return TextIO;
} );
