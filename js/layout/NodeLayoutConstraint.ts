// Copyright 2021-2022, University of Colorado Boulder

/**
 * Supertype for LayoutConstraints that are based on an actual Node where the layout takes place
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import { GridConfigurableOptions, LayoutConstraint, Node, scenery } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import Tandem from '../../../tandem/js/Tandem.js';

type SelfOptions = {
  // Whether invisible Nodes are excluded from the layout.
  excludeInvisible?: boolean;

  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;
};

export type NodeLayoutConstraintOptions = SelfOptions & GridConfigurableOptions;

// Type export designed for use with clients
export type NodeLayoutAvailableConstraintOptions = Pick<NodeLayoutConstraintOptions, 'excludeInvisible'>;

export default class NodeLayoutConstraint extends LayoutConstraint {

  private _excludeInvisible = true;

  // Reports out the used layout bounds (may be larger than actual bounds, since it will include margins, etc.)
  readonly layoutBoundsProperty: IProperty<Bounds2>;

  readonly preferredWidthProperty: IProperty<number | null>;
  readonly preferredHeightProperty: IProperty<number | null>;
  readonly minimumWidthProperty: IProperty<number | null>;
  readonly minimumHeightProperty: IProperty<number | null>;

  constructor( ancestorNode: Node, providedOptions?: NodeLayoutConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    // The omitted options are set to proper defaults below
    const options = optionize<NodeLayoutConstraintOptions, Omit<SelfOptions, 'excludeInvisible' | 'spacing' | 'xSpacing' | 'ySpacing'>, GridConfigurableOptions>()( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty<number | null>( null ),
      preferredHeightProperty: new TinyProperty<number | null>( null ),
      minimumWidthProperty: new TinyProperty<number | null>( null ),
      minimumHeightProperty: new TinyProperty<number | null>( null )
    }, providedOptions );

    super( ancestorNode );

    this.layoutBoundsProperty = new Property( Bounds2.NOTHING, {
      useDeepEquality: true,
      tandem: Tandem.OPT_OUT
    } );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
  }

  get excludeInvisible(): boolean {
    return this._excludeInvisible;
  }

  set excludeInvisible( value: boolean ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisible !== value ) {
      this._excludeInvisible = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * Releases references
   */
  override dispose(): void {
    // In case they're from external sources
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );

    super.dispose();
  }
}

scenery.register( 'NodeLayoutConstraint', NodeLayoutConstraint );
