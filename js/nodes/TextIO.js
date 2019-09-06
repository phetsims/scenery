// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for Text
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var FontIO = require( 'SCENERY/util/FontIO' );
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );
  var NodeProperty = require( 'SCENERY/util/NodeProperty' );
  var NumberIO = require( 'TANDEM/types/NumberIO' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var scenery = require( 'SCENERY/scenery' );
  var StringIO = require( 'TANDEM/types/StringIO' );
  var VoidIO = require( 'TANDEM/types/VoidIO' );

  class TextIO extends NodeIO {

    /**
     * @param {Text} text
     * @param {string} phetioID
     * @constructor
     */
    constructor( text, phetioID ) {
      super( text, phetioID );

      // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
      var textProperty = new NodeProperty( text, 'text', 'text', _.extend( {

        // pick the following values from the parent Node
        phetioReadOnly: text.phetioReadOnly,
        phetioState: true,
        phetioType: PropertyIO( StringIO ),

        tandem: text.tandem.createTandem( 'textProperty' ),
        phetioDocumentation: 'Property for the displayed text'
      }, text.phetioComponentOptions, text.phetioComponentOptions.textProperty ) );

      // @private
      this.disposeTextIO = function() {
        textProperty.dispose();
      };
    }

    /**
     * @public - called by PhetioObject when the wrapper is done
     */
    dispose() {
      this.disposeTextIO();
      NodeIO.prototype.dispose.call( this );
    }
  }

  TextIO.methods = {
    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ FontIO ],
      implementation: function( font ) {
        this.phetioObject.setFont( font );
      },
      documentation: 'Sets font options for this TextIO instance, e.g. {size: 16, weight: bold}. If increasing the font ' +
                     'size does not make the text size larger, you may need to increase the maxWidth of the TextIO also.',
      invocableForReadOnlyElements: false
    },

    getFontOptions: {
      returnType: FontIO,
      parameterTypes: [],
      implementation: function() {
        return this.phetioObject.getFont();
      },
      documentation: 'Gets font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( maxWidth ) {
        this.phetioObject.setMaxWidth( maxWidth );
      },
      documentation: 'Sets the maximum width of text box. ' +
                     'If the text width exceeds maxWidth, it is scaled down to fit.',
      invocableForReadOnlyElements: false
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function() {
        return this.phetioObject.maxWidth;
      },
      documentation: 'Gets the maximum width of text box'
    }
  };
  TextIO.documentation = 'Text that is displayed in the simulation. TextIO has a nested PropertyIO.&lt;String&gt; for ' +
                         'the current string value.';
  TextIO.validator = { valueType: scenery.Text };
  TextIO.typeName = 'TextIO';
  ObjectIO.validateSubtype( TextIO );

  return scenery.register( 'TextIO', TextIO );
} );
