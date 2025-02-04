// Copyright 2021-2025, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for FlowConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import type { ExternalFlowConfigurableOptions } from '../../layout/constraints/FlowConfigurable.js';
import { FLOW_CONFIGURABLE_OPTION_KEYS } from '../../layout/constraints/FlowConfigurable.js';
import FlowConfigurable from '../../layout/constraints/FlowConfigurable.js';
import type FlowConstraint from '../../layout/constraints/FlowConstraint.js';
import LayoutAlign from '../../layout/LayoutAlign.js';
import type LayoutProxy from '../../layout/LayoutProxy.js';
import MarginLayoutCell from '../../layout/constraints/MarginLayoutCell.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';

const FLOW_CELL_KEYS = [
  ...FLOW_CONFIGURABLE_OPTION_KEYS,
  'isSeparator'
];

type SelfOptions = {
  // Allows marking a cell as a "separator", such that multiple adjacent separators (and those at the start/end) get
  // collapsed (all but the first are not included in layout AND made invisible)
  isSeparator?: boolean;
};

export type FlowCellOptions = SelfOptions & StrictOmit<ExternalFlowConfigurableOptions, 'orientation'>;

export default class FlowCell extends FlowConfigurable( MarginLayoutCell ) {

  // (scenery-internal) Set during FlowConstraint layout
  public size = 0;

  // (scenery-internal)
  public _isSeparator = false;

  private readonly flowConstraint: FlowConstraint;

  public constructor( constraint: FlowConstraint, node: Node, proxy: LayoutProxy | null ) {
    super( constraint, node, proxy );

    this.flowConstraint = constraint;

    this.orientation = constraint.orientation;
    this.onLayoutOptionsChange( true );
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  public get effectiveAlign(): LayoutAlign {
    return this._align !== null ? this._align : this.flowConstraint._align!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  public get effectiveCellAlign(): LayoutAlign {
    return this._cellAlign !== null ? this._cellAlign : this.flowConstraint._cellAlign!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  public get effectiveStretch(): boolean {
    return this._stretch !== null ? this._stretch : this.flowConstraint._stretch!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  public get effectiveGrow(): number {
    return this._grow !== null ? this._grow : this.flowConstraint._grow!;
  }

  protected override onLayoutOptionsChange( isInitial = false ): void {
    if ( this.node.layoutOptions ) {
      this.setOptions( this.node.layoutOptions as ExternalFlowConfigurableOptions );
    }

    super.onLayoutOptionsChange();

    if ( !isInitial ) {
      this.flowConstraint.updateLayoutAutomatically();
    }
  }

  private setOptions( providedOptions?: ExternalFlowConfigurableOptions ): void {

    const options = optionize<FlowCellOptions, SelfOptions, ExternalFlowConfigurableOptions>()( {
      isSeparator: false
    }, providedOptions );

    assert && Object.keys( options ).forEach( key => {
      assert && assert( FLOW_CELL_KEYS.includes( key ), `Cannot provide key ${key} to a FlowCell's layoutOptions. Perhaps this is a Grid-style layout option?` );
    } );

    this._isSeparator = options.isSeparator;

    // Clear options that are not provided in the mutate options (but if they are present, do NOT clear, so we avoid
    // more potential for infinite loops)
    this.setConfigToInherit( options );

    this.mutateConfigurable( options );
  }
}

scenery.register( 'FlowCell', FlowCell );