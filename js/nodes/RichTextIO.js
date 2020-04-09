// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO type for RichText
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import PropertyIO from '../../../axon/js/PropertyIO.js';
import merge from '../../../phet-core/js/merge.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import scenery from '../scenery.js';
import NodeProperty from '../util/NodeProperty.js';
import NodeIO from './NodeIO.js';

class RichTextIO extends NodeIO {
  /**
   * @param {RichText} richText
   * @param {string} phetioID
   */
  constructor( richText, phetioID ) {
    super( richText, phetioID );

    // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
    const textProperty = new NodeProperty( richText, richText.textProperty, 'text', merge( {

      // pick the following values from the parent Node
      phetioReadOnly: richText.phetioReadOnly,
      phetioType: PropertyIO( StringIO ),

      tandem: richText.tandem.createTandem( 'textProperty' ),
      phetioDocumentation: 'Property for the displayed text'
    }, richText.phetioComponentOptions, richText.phetioComponentOptions.textProperty ) );

    // @private
    this.disposeRichTextIO = function() {
      textProperty.dispose();
    };
  }

  /**
   * @public - called by PhetioObject when the wrapper is done
   */
  dispose() {
    this.disposeRichTextIO();
    NodeIO.prototype.dispose.call( this );
  }
}

RichTextIO.documentation = 'The tandem IO type for the scenery RichText node';
RichTextIO.validator = { valueType: scenery.RichText };
RichTextIO.typeName = 'RichTextIO';
ObjectIO.validateSubtype( RichTextIO );

scenery.register( 'RichTextIO', RichTextIO );
export default RichTextIO;